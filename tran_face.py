import options.opt as opt
import torch
import data.face_dataset as face_dataset
from torch.utils.data import DataLoader
import numpy as np
import models.audio2feature as trans
import torch.nn.functional as F
import os

import warnings

warnings.filterwarnings('ignore')

args = opt.get_args_parser()
print(args)

import datetime
current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
print(current_time)

run_name = f'tran_face-{current_time}'
path = f'ckpt/tran_face/{run_name}'
os.makedirs(path, exist_ok=True)
if not os.path.exists(path):
    os.mkdir(path)

model = trans.Audio2faceFeature_Transformer(args)

if args.resume_trans:
    print('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    model.load_state_dict(ckpt, strict=True)
model.train()
model.to(device=args.device)

params = [p for p in model.parameters() if p.requires_grad is True]

optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=10000, max_lr=args.learning_rate, pct_start=0.1, div_factor=25, final_div_factor=1 / 25, anneal_strategy='linear')

train_dataset = face_dataset.FaceAudioDataset(args) if args.update_audio else face_dataset.FaceAudioEmbDataset(args)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_cond, num_workers=args.num_workers, shuffle=True)

val_dataset = face_dataset.FaceAudioDataset(args) if args.update_audio else face_dataset.FaceAudioEmbDataset(args, True)
val_dataloader = DataLoader(val_dataset, batch_size=256, num_workers=args.num_workers)

all_indices = torch.arange(0, args.input_dim_lip+args.input_dim_face)
lip_indices = torch.tensor([0,1,2,3,4,5,7,9,10,11,12,13,14])
mask = ~torch.isin(all_indices, lip_indices)
face_indices = all_indices[mask]

min_dis = 999

for epoch in range(args.epochs):
    train_face_loss, train_rec_loss, train_af_contra, train_f_contra = [], [], [], []
    for batch in train_dataloader:
        info, coeff_3dmm, audio, ref = batch
        coeff_3dmm = coeff_3dmm.to(device=args.device)
        ref = ref.to(device=args.device)[:, face_indices]
        gt_coeff_face = coeff_3dmm[:, :, face_indices]
        audio = audio.to(device=args.device, dtype=torch.float)

        face_code_idx = model.vqgan.encode(gt_coeff_face)
        face_codebook = model.vqgan.vqvae.face_quantizer.codebook.data.detach().clone()
        quantized_face = face_codebook[face_code_idx]
        out = model(quantized_face, audio, ref)
        out_face = out['out_face']
        face_loss = F.mse_loss(out_face, quantized_face)
        if args.contra > 0:
            f_contra_loss = F.cross_entropy(out['f_logits'], face_code_idx.view(-1))
        loss = face_loss + f_contra_loss * args.contra if args.contra > 0 else face_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            out_face = model.vqgan.vqvae.preprocess(out_face)
            decoded_face = model.vqgan.vqvae.face_decoder(out_face)
            rec_face = model.vqgan.vqvae.postprocess(decoded_face)
            train_rec_loss.append(F.mse_loss(rec_face, gt_coeff_face).cpu())

        train_face_loss.append(face_loss.item())
        if args.contra > 0:
            train_f_contra.append(f_contra_loss.item())

    msg = f"Train. Epoch {epoch}: face_loss. {np.mean(train_face_loss):.4f} rec_loss: {np.mean(train_rec_loss):.4f}"
    
    print(msg, flush=True)

    if epoch % 10 == 0:
        model.eval()
        rec_vq, rec_tran, val_face_loss = [], [], []
        for batch in val_dataloader:
            info, coeff_3dmm, audio, ref = batch
            coeff_3dmm = coeff_3dmm.to(device=args.device)
            ref = ref.to(device=args.device)[:, face_indices]
            gt_coeff_face = coeff_3dmm[:, :, face_indices]
            audio = audio.to(device=args.device, dtype=torch.float)

            with torch.no_grad():
                face_code_idx = model.vqgan.encode(gt_coeff_face)
                face_codebook = model.vqgan.vqvae.face_quantizer.codebook.data.detach().clone()
                quantized_face = face_codebook[face_code_idx]
                out = model(quantized_face, audio, ref)
                out_face = out['out_face']
                face_loss = F.mse_loss(out_face, quantized_face)
                
                x_rec = model.vqgan.vqvae.forward_decoder(face_code_idx)
                out_face = model.vqgan.vqvae.preprocess(out_face)
                decoded_face = model.vqgan.vqvae.face_decoder(out_face)
                rec_face = model.vqgan.vqvae.postprocess(decoded_face)

                rec_tran.append(F.mse_loss(gt_coeff_face, rec_face).cpu().numpy())
                rec_vq.append(F.mse_loss(gt_coeff_face, x_rec).cpu().numpy())
                val_face_loss.append(face_loss.item())

        msg = f"Val. val_rec_vq. {np.mean(rec_vq):.4f} face_loss. {np.mean(val_face_loss):.4f}, rec_tran. {np.mean(rec_tran):.4f}"
        print(msg, flush=True)

        if min_dis > np.mean(rec_tran) and np.mean(rec_tran) < 0.04:
            min_dis = np.mean(rec_tran)
            torch.save(model.state_dict(), f"{path}/tran.pt")
            print(f"{path}/tran.pt", flush=True)
        model.train()
        