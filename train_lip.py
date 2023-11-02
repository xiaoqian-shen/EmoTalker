import options.opt as opt
import torch
import data.lip_dataset as face_dataset
from torch.utils.data import DataLoader
import numpy as np
import models.audio_encoder as encoder
import torch.nn.functional as F
import os
import warnings
import torch.nn as nn
from tqdm import tqdm

warnings.filterwarnings('ignore')

args = opt.get_args_parser()
print(args)

import datetime
current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
print(current_time)
run_name = f'audio2lip-{current_time}'
path = f'ckpt/audio2lip/{run_name}'
os.makedirs(path, exist_ok=True)
if not os.path.exists(path):
    os.mkdir(path)

speech_encoder = encoder.SimpleWrapperV2(args)

if args.resume_trans:
    print('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    speech_encoder.load_state_dict(ckpt, strict=True)
speech_encoder.train()
speech_encoder.to(device=args.device)

params = [p for p in speech_encoder.parameters() if p.requires_grad is True]

optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=args.weight_decay)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

train_dataset = face_dataset.FaceAudioEmbDataset(args)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_cond, num_workers=args.num_workers, shuffle=True)

val_dataset = face_dataset.FaceAudioEmbDataset(args, True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size_cond, num_workers=args.num_workers)
lip_indices = torch.tensor([0,1,2,3,4,5,7,9,10,11,12,13,14])

min_dis = 999

for epoch in range(args.epochs):
    train_rec_loss = []
    for batch in train_dataloader:
        info, coeff_3dmm, audio, ref = batch
        ref = ref.to(device=args.device)[:, :64]
        coeff_3dmm = coeff_3dmm.to(device=args.device)
        gt_lip = coeff_3dmm[:, :, lip_indices]
        audio = audio.to(device=args.device, dtype=torch.float)
        pred_lip = speech_encoder(audio, ref)
        loss = F.mse_loss(pred_lip, coeff_3dmm[:,:,lip_indices])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        train_rec_loss.append(loss.item())

    msg = f"Train. Epoch {epoch}: rec_loss. {np.mean(train_rec_loss):.4f}"
    print(msg, flush=True)

    speech_encoder.eval()
    rec_vq, rec_tran, val_quan_loss = [], [], []
    for batch in val_dataloader:
        info, coeff_3dmm, audio, ref = batch
        coeff_3dmm = coeff_3dmm.to(device=args.device)
        audio = audio.to(device=args.device, dtype=torch.float)
        ref = ref.to(device=args.device)[:, :64]
        gt_lip = coeff_3dmm[:, :, lip_indices]
        with torch.no_grad():
            pred_lip = speech_encoder(audio, ref)
            # val_loss = F.mse_loss(pred_lip[:, :, lip_indices], gt_lip)
            val_loss = F.mse_loss(pred_lip, coeff_3dmm[:,:, lip_indices])
            rec_tran.append(val_loss.cpu().numpy())

    msg = f"Val. rec_tran. {np.mean(rec_tran):.4f}"
    print(msg, flush=True)
    speech_encoder.train()

    if min_dis > np.mean(rec_tran):
        min_dis = np.mean(rec_tran)
        torch.save(speech_encoder.state_dict(), f"{path}/audio2lip.pt")
        print(f"{path}/audio2lip.pt", flush=True)

    
