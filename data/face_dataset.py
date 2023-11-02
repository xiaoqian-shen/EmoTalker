import numpy as np
from scipy.io import loadmat
from collections import defaultdict

import json
import pickle
import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

emotions = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
levels = ['level_1', 'level_2', 'level_3']

class FaceDataset(Dataset):
    def __init__(self, args, val = False):
        self.root_path = args.path
        with open('data/split.json','r') as f:
            self.data = json.load(f)
            self.filenames = self.data['val']['coeff'] if val else self.data['train']['coeff']
        self.clip_len = args.clip_len

    def get_3dmm(self, file_path):
        file_mat = loadmat(file_path)
        coeff_3dmm = file_mat['coeff']
        crop_param = file_mat['transform_params']
        _, _, ratio, t0, t1 = np.hsplit(crop_param.astype(np.float32), 5)
        crop_param = np.concatenate([ratio, t0, t1], 1)
        coeff_3dmm_cat = np.concatenate([coeff_3dmm, crop_param], 1)
        return coeff_3dmm_cat

    def get_video_name(self, file_path):
        file_path = file_path.split('/')[-1]
        splits = file_path.split('.')[0].split('-')
        return splits[0], splits[1], splits[2], splits[3]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        file_path = self.filenames[index]
        p_id, emotion, level, v_id = self.get_video_name(self.filenames[index])
        info = {'p_id': p_id, 'emotion': emotion, 'level': level, 'v_id': v_id}
        coeff_3dmm = self.get_3dmm(file_path)
        coeff_3dmm = self.transform_semantic(coeff_3dmm)
        if coeff_3dmm.shape[0] <= self.clip_len:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, coeff_3dmm.shape[0] - self.clip_len)
        coeff_3dmm = coeff_3dmm[start_idx: start_idx + self.clip_len]
        emotion_label = emotions.index(emotion)
        level_label = levels.index(level) + 1
        gt_emotion = torch.Tensor([emotion_label])

        return gt_emotion, coeff_3dmm

    def transform_semantic(self, coeff_3dmm):
        rest = defaultdict()
        rest['id_coeff'] = coeff_3dmm[:,:80] #identity
        ex_coeff = coeff_3dmm[:, 80:144]  # expression
        rest['tex_coeff'] = coeff_3dmm[:,144:224] # texture
        angles = coeff_3dmm[:, 224:227]  # euler angles for pose
        rest['gamma'] = coeff_3dmm[:,227:254] #lighting
        translation = coeff_3dmm[:, 254:257]  # translation
        rest['crop'] = coeff_3dmm[:, 257:260]  # crop param

        coeff_3dmm = np.concatenate([ex_coeff, angles, translation], 1)
        return torch.Tensor(coeff_3dmm)
 
class FaceAudioEmbDataset(Dataset):
    def __init__(self, args, val = False):
        self.root_path = args.path
        with open('data/split.json','r') as f:
            self.data = json.load(f)
            self.filenames = self.data['val']['coeff'] if val else self.data['train']['coeff']
        self.clip_len = args.clip_len
        self.args = args
        with open('data/audio_hubert.pkl', 'rb') as f:
            self.audio = pickle.load(f)
        self.val = val
        
    def get_3dmm(self, file_path):
        file_mat = loadmat(file_path)
        coeff_3dmm = file_mat['coeff']
        crop_param = file_mat['transform_params']
        _, _, ratio, t0, t1 = np.hsplit(crop_param.astype(np.float32), 5)
        crop_param = np.concatenate([ratio, t0, t1], 1)
        coeff_3dmm_cat = np.concatenate([coeff_3dmm, crop_param], 1)
        return coeff_3dmm_cat

    def get_video_info(self, file_path):
        splits = file_path.split('.')[0].split('data/mead-3dmm/')[-1].split('-')
        return splits[0], splits[1], splits[2], splits[3]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        # 3dmm
        file_3dmm = self.filenames[index]
        p_id, emotion, level, v_id = self.get_video_info(file_3dmm)
        info = {'p_id': p_id, 'emotion': emotion, 'level': level, 'v_id': v_id}
        coeff_3dmm = self.get_3dmm(file_3dmm)
        coeff_3dmm = self.transform_semantic(coeff_3dmm)
        audio = self.audio[file_3dmm].detach().cpu().squeeze()

        start_idx = 0 if coeff_3dmm.shape[0] <= self.clip_len else np.random.randint(0, coeff_3dmm.shape[0] - self.clip_len)
        coeff_3dmm = coeff_3dmm[start_idx: start_idx + self.clip_len]

        start_idx = np.ceil(start_idx * (191.0 / 96.0)).astype(int)
        if start_idx > audio.shape[0] - 191:
            start_idx = audio.shape[0] - 191
        audio = audio[start_idx: start_idx + 191]
        
        ref_idx = np.random.randint(0, coeff_3dmm.shape[0])
        ref = coeff_3dmm[ref_idx]
        
        return info, coeff_3dmm, audio, ref

    def transform_semantic(self, coeff_3dmm):
        rest = defaultdict()
        rest['id_coeff'] = coeff_3dmm[:,:80] #identity
        ex_coeff = coeff_3dmm[:, 80:144]  # expression
        rest['tex_coeff'] = coeff_3dmm[:,144:224] # texture
        angles = coeff_3dmm[:, 224:227]  # euler angles for pose
        rest['gamma'] = coeff_3dmm[:,227:254] # lighting
        translation = coeff_3dmm[:, 254:257]  # translation
        rest['crop'] = coeff_3dmm[:, 257:260]  # crop param

        coeff_3dmm = np.concatenate([ex_coeff, angles, translation], 1)
        return torch.Tensor(coeff_3dmm)