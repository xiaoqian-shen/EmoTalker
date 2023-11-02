import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os
import numpy as np
import torch.nn.functional as F

from models.utils import WarmupLinearLRSchedule

import data.face_dataset as face_dataset
import options.opt as opt
import models.vq_face as vq

import datetime
current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
print(current_time)
args = opt.get_args_parser()

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

print(args)

run_name = f'vq_face-{current_time}'
path = f'ckpt/vq_face/{run_name}'
os.makedirs(path, exist_ok=True)
if not os.path.exists(path):
    os.mkdir(path)

if args.wandb:
    wandb.init(project="emotalker", name=run_name, config=vars(args))

model = vq.FaceVQVAE(args)

train_dataset = face_dataset.FaceDataset(args)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

val_dataset = face_dataset.FaceDataset(args, True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

if args.vq_ckpt is not None and args.resume_vq:
    print('load')
    ckpt = torch.load(args.vq_ckpt, map_location='cpu')
    model.load_state_dict(ckpt, strict=True)
model.train()
model.cuda()

# ---- Optimizer & Scheduler ---- #
optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

RecLoss = torch.nn.SmoothL1Loss()
CELoss = torch.nn.CrossEntropyLoss()

all_indices = torch.arange(0, 70)
lip_indices = torch.tensor([0,1,2,3,4,5,7,9,10,11,12,13,14])
mask = ~torch.isin(all_indices, lip_indices)
face_indices = all_indices[mask]
min_dis = 9999

for nb_iter in range(1, args.warm_up_iter):
    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
    for batch in train_dataloader:
        gt_emotion, gt_coeff = batch
        gt_coeff = gt_coeff.cuda().float()
        gt_coeff_face = gt_coeff[:, :, face_indices]
        gt_emotion = torch.Tensor(gt_emotion).view(-1).to(dtype=torch.long).cuda()
        out = model(gt_coeff_face)
        loss_face = RecLoss(out['rec_face'], gt_coeff_face)
        if args.cls_num > 0:
            train_emotion_loss = CELoss(out['logits'], gt_emotion)
        
        loss = loss_face + args.commit * out['face_loss']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print("epoch: {} train_rec_loss: {:.4f} commit_loss: {:.4f}".format(
    #     nb_iter, train_rec_loss.item(), out['lip_loss'].item()+out['face_loss'].item()), flush=True)

print('training...', flush=True)

##### ---- Training ---- #####
nb_iter = 0
for epoch in range(args.epochs):
    for batch in train_dataloader:
        gt_emotion, gt_coeff = batch
        gt_coeff = gt_coeff.cuda().float()
        gt_coeff_face = gt_coeff[:, :, face_indices]
        gt_emotion = torch.Tensor(gt_emotion).view(-1).to(dtype=torch.long).cuda()

        out = model(gt_coeff_face)
        loss_face = RecLoss(out['rec_face'], gt_coeff_face)
        if args.cls_num > 0:
            train_emotion_loss = CELoss(out['logits'], gt_emotion)

        loss = loss_face + args.commit * out['face_loss'] + args.emotion * train_emotion_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if args.wandb:
            wandb.log({
                        "train_face_rec_loss": loss_face.item(),
                        "train_emo_loss": train_emotion_loss.item(),
                        "face_commit": out['face_loss'].item(),
                        "lip_commit": out['lip_loss'].item(),
                        "lr": current_lr,
                    })

    val_rec_loss = []
    val_emo_loss = []
    
    with torch.no_grad():
        model.eval()
        for batch in val_dataloader:
            gt_emotion, gt_coeff = batch
            gt_emotion = torch.Tensor(gt_emotion).view(-1).to(dtype=torch.long).cuda()
            gt_coeff = gt_coeff.cuda().float()
            gt_coeff_face = gt_coeff[:, :, face_indices]
            out = model(gt_coeff_face)
            val_rec_loss.append(F.mse_loss(out['rec_face'], gt_coeff_face).cpu().numpy())
            if args.cls_num > 0:
                val_emo_loss.append(CELoss(out['logits'], gt_emotion).cpu().numpy())
            
        print("epoch: {} train_rec_face: {:.4f} train_commit_face: {:.4f} val_rec_loss: {:.4f}".format(
            epoch, loss_face.item(), out['face_loss'].item(), np.mean(val_rec_loss)), flush=True)
        model.train()

    if args.wandb:
        wandb.log({
            "val_rec_loss": np.mean(val_rec_loss),
            "val_emo_loss": np.mean(val_emo_loss),
        })

    if min_dis > np.mean(val_rec_loss) and np.mean(val_rec_loss) < 0.004:
        min_dis = np.mean(val_rec_loss)
        torch.save(model.state_dict(), f"{path}/vq.pt")
        print(f"{path}/vq.pt", flush=True)