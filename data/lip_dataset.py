import numpy as np
from scipy.io import loadmat
import librosa
from collections import defaultdict
# from pydub import AudioSegment
import os
import random
import math
import PIL.Image
import json
import pickle
from models.utils import *

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

emotions = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
levels = ['level_1', 'level_2', 'level_3']

syncnet_T = 5
syncnet_mel_step_size = 16

class FaceAudioEmbDataset(Dataset):
    def __init__(self, args, val = False):
        self.root_path = args.path
        with open('data/split.json','r') as f:
            self.data = json.load(f)
            self.filenames = self.data['val']['coeff'] if val else self.data['train']['coeff']
        self.clip_len = args.clip_len
        self.args = args
        with open('data/audio_mel.pkl', 'rb') as f:
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
        splits = file_path.split('.')[0].split('data/mead-3dmm')[-1].split('-')
        return splits[0], splits[1], splits[2], splits[3]

    def __len__(self):
        return len(self.filenames)

    def crop_audio_window(self, spec, start_frame_num):

        start_idx = int(80. * (start_frame_num / float(25)))
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, faces, spec):
        start_frame_num = random.randint(0, faces.shape[0] - 15) + 2
        face = faces[start_frame_num: start_frame_num + syncnet_T, :]
        mels = []
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            mels.append(m.T)
        mels = np.asarray(mels)
        index = random.sample(range(0, faces.shape[0]), 1)[0]
        ref = faces[index]
        
        return face, mels, ref

    def __getitem__(self, index):
        file_3dmm = self.filenames[index]
        p_id, emotion, level, v_id = self.get_video_info(file_3dmm)
        info = {'p_id': p_id, 'emotion': emotion, 'level': level, 'v_id': v_id}
        coeff_3dmm = self.get_3dmm(file_3dmm)
        coeff_3dmm = self.transform_semantic(coeff_3dmm)
        audio = self.audio[file_3dmm].detach().cpu().numpy()
        coeff_3dmm, audio, ref = self.get_segmented_mels(coeff_3dmm, audio)
        audio = torch.from_numpy(audio)

        return info, coeff_3dmm, audio, ref

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