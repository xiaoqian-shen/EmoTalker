import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser
import scipy.io as scio
import pickle
import random

import options.render as opt
import torch
import data.face_dataset as face_dataset
from torch.utils.data import DataLoader
import numpy as np
import models.audio2feature as trans
from scipy.io import savemat
import os
import models.audio_encoder as audio_encoder
from models.utils import *

from renderer.src.utils.preprocess import CropAndExtract
from renderer.src.facerender.animate import AnimateFromCoeff
from renderer.src.generate_facerender_batch import get_facerender_data

import warnings
warnings.filterwarnings('ignore')
args = opt.get_args_parser()

syncnet_mel_step_size = 16
fps = 25

trans_encoder = trans.Audio2faceFeature_Transformer(args)
run_name = '2023_05_09_21_07_12'
ckpt_path = f'ckpt/tran_face/tran_face-{run_name}/tran.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')
trans_encoder.load_state_dict(ckpt, strict=True)

trans_encoder.to(device=args.device)
trans_encoder.eval()

train_dataset = face_dataset.FaceAudioMelTest(args)
train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=args.num_workers)

audio2lip = audio_encoder.SimpleWrapperV2(args).to(device=args.device)

finetune_ckpt = 'ckpt/audio2lip/audio2lip-2023_05_09_08_58_44/audio2lip.pt'
audio2lip.load_state_dict(torch.load(finetune_ckpt, map_location='cpu'))

audio2lip.eval()
all_indices = torch.arange(0, 70)
lip_indices = torch.tensor([0,1,2,3,4,5,7,9,10,11,12,13,14])
mask = ~torch.isin(all_indices, lip_indices)
face_indices = all_indices[mask]

with open('data/audio_hubert.pkl', 'rb') as f:
    audio_hubert_pkl = pickle.load(f)
with open('data/audio_mel.pkl', 'rb') as f:
    audio_mel_pkl = pickle.load(f)

def generate_blink_seq_randomly(num_frames):
    ratio = np.zeros((num_frames,1))
    if num_frames<=20:
        return ratio
    frame_id = 0
    while frame_id in range(num_frames):
        start = random.choice(range(min(10,num_frames), min(int(num_frames/2), 70))) 
        if frame_id+start+5<=num_frames - 1:
            ratio[frame_id+start:frame_id+start+5, 0] = [0.5, 0.9, 1.0, 0.9, 0.5]
            frame_id = frame_id+start+5
        else:
            break
    return ratio

def main(args):
    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir)
    os.makedirs(save_dir, exist_ok=True)
    device = args.device
    batch_size = args.batch_size
    input_yaw_list = args.input_yaw
    input_pitch_list = args.input_pitch
    input_roll_list = args.input_roll

    current_code_path = sys.argv[0]
    current_root_path = os.path.split(current_code_path)[0] + 'renderer'

    os.environ['TORCH_HOME']=os.path.join(current_root_path, args.checkpoint_dir)

    path_of_lm_croper = os.path.join(current_root_path, args.checkpoint_dir, 'shape_predictor_68_face_landmarks.dat')
    path_of_net_recon_model = os.path.join(current_root_path, args.checkpoint_dir, 'epoch_20.pth')
    dir_of_BFM_fitting = os.path.join(current_root_path, args.checkpoint_dir, 'BFM_Fitting')

    free_view_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'facevid2vid_00189-model.pth.tar')

    if args.preprocess == 'full':
        mapping_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'mapping_00109-model.pth.tar')
        facerender_yaml_path = os.path.join(current_root_path, 'src', 'config', 'facerender_still.yaml')
    else:
        mapping_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'mapping_00229-model.pth.tar')
        facerender_yaml_path = os.path.join(current_root_path, 'src', 'config', 'facerender.yaml')

    #init model
    print(path_of_net_recon_model)
    preprocess_model = CropAndExtract(path_of_lm_croper, path_of_net_recon_model, dir_of_BFM_fitting, device)

    animate_from_coeff = AnimateFromCoeff(free_view_checkpoint, mapping_checkpoint, 
                                            facerender_yaml_path, device)

    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, args.preprocess, source_image_flag=(not args.random))
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return
    
    source_semantics = torch.tensor(scio.loadmat(first_coeff_path)['coeff_3dmm'])[:, :70]

    #audio2coeff
    with torch.no_grad():
        file_3dmm = f'data/mead-3dmm/{args.p_id}-{args.emotion}-{args.level}-{args.v_id}.mat'
        audio = audio_hubert_pkl[file_3dmm][0].to(device=args.device, dtype=torch.float).unsqueeze(0)
        audio_mel = audio_mel_pkl[file_3dmm].detach().cpu().to(dtype=torch.float).squeeze()

        ref = source_semantics[:, face_indices].to(device=args.device, dtype=torch.float)
        out_feature = trans_encoder.sample(audio, ref)
        out_feature = trans_encoder.vqgan.vqvae.preprocess(out_feature)
        x_decoder = trans_encoder.vqgan.vqvae.face_decoder(out_feature.detach())
        face = trans_encoder.vqgan.vqvae.postprocess(x_decoder)

        wav = load_wav(audio_path, 16000)
        wav_length, num_frames = parse_audio_length(len(wav), 16000, 25)

        spec = np.asarray(audio_mel).copy()
        indiv_mels = []

        for i in range(num_frames):
            start_frame_num = i-2
            start_idx = int(80. * (start_frame_num / float(fps)))
            end_idx = start_idx + syncnet_mel_step_size
            seq = list(range(start_idx, end_idx))
            seq = [ min(max(item, 0), audio_mel.shape[0]-1) for item in seq ]
            m = spec[seq, :]
            indiv_mels.append(m.T)
        mel_input = torch.tensor(np.asarray(indiv_mels)).unsqueeze(0)       # 1 T 80 16
        bs = mel_input.shape[0]
        T = mel_input.shape[1]
        face_t = face.squeeze().shape[0]
        if face_t < T:
            num_repeats = T - face_t
            last_row = face.squeeze()[-1:]
            extended_face = np.tile(last_row, (num_repeats, 1))
            extended_face = np.concatenate((face.squeeze(), extended_face), axis=0)
        else:
            extended_face = face
        exp_coeff_pred = []
        random_ratio = generate_blink_seq_randomly(T)
        for i in range(0, T, 10):
            current_mel_input = mel_input[:,i:i+10]
            audiox = current_mel_input.view(-1, 1, 80, 16).to(device=args.device, dtype=torch.float)
            ref = source_semantics[:, :64].to(device=args.device, dtype=torch.float)
            ref = ref.repeat_interleave(audiox.size(0), dim=0)
            ref[:, face_indices[:51]] = extended_face[:,i:i+ref.shape[0]][:, :, :51]
            ratio = torch.tensor(random_ratio[i:i+10]).to(device=args.device, dtype=torch.float)
            curr_exp_coeff_pred  = audio2lip.test_freeze_linear(audiox, ref, ratio)
            exp_coeff_pred += [curr_exp_coeff_pred]
        exp_coeff_pred = torch.cat(exp_coeff_pred, axis=0)
        mouth  = exp_coeff_pred
        
        face, mouth = face.squeeze(), mouth.squeeze()
        gen_len = min(face.shape[0], mouth.shape[0])
        face = face[:gen_len]
        mouth = mouth[:gen_len]
        x_new = torch.empty(gen_len, 70).cpu()
        x_new[:, lip_indices] = mouth.cpu()
        x_new[:, face_indices] = face.cpu()

        gen_len = x_new.shape[0]
        ex_coeff = x_new[:, 0: 64].cpu().numpy()
        angles = x_new[:, 64: 67].cpu().numpy()
        translation = x_new[:, 67: 70].cpu().numpy()
        
        coeff_3dmm = np.concatenate([ex_coeff, angles, translation], 1)
        
        name = f"{args.p_id}-{args.emotion}-{args.level}-{args.v_id}.mat"
        save_path = f'3dmm_results/{args.save_name}'
        os.makedirs(save_path, exist_ok=True)
        args.pred_coeff_path = os.path.join(save_path, name)
        savemat(args.pred_coeff_path, {'coeff':coeff_3dmm})

    #coeff2video
    data = get_facerender_data(args.pred_coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess)
    
    animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                enhancer=args.enhancer, background_enhancer=args.background_enhancer, preprocess=args.preprocess)
    
if __name__ == '__main__':

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"

    import json
    if args.select:
        with open('data/mead_select.json','r') as f:
            data = json.load(f)
    else:
        with open('data/split.json','r') as f:
            data = json.load(f)['test']['coeff']
    with open('data/neutral.json','r') as f:
        neutral_dict = json.load(f)
    for d in data:
        filename = d.split('/')[-1]
        words = filename.split('.')[0].split('-')
        p_id = words[0]
        emotion = words[1]
        level = words[2]
        v_id = words[3]
        if args.neutral:
            if p_id in neutral_dict.keys():
                neutral_filename = neutral_dict[p_id]
                neutral_words = neutral_filename.split('-')
                args.source_image = os.path.join(f'data/MEAD/{p_id}/video/front/{neutral_words[1]}/{neutral_words[2]}/{neutral_words[3]}.mp4')
            else:
                args.source_image = os.path.join(f'data/MEAD/{p_id}/video/front/{emotion}/level_1/001.mp4')
        else:
            args.source_image = os.path.join(f'data/MEAD/{p_id}/video/front/{emotion}/{level}/001.mp4')
        args.driven_audio = os.path.join(f'data/MEAD/{p_id}/audio/{emotion}/{level}', f'{v_id}.wav')
        args.pred_coeff_path = os.path.join(f'3dmm_results/{args.save_name}', filename)
        args.p_id = p_id
        args.emotion = emotion
        args.level = level
        args.v_id = v_id
        if os.path.exists(args.source_image) and os.path.exists(args.driven_audio):
            args.result_dir = f'examples/{args.save_name}-{args.prefix}'
            main(args)