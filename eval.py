import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser

from PIL import Image
import cv2

from torchvision import transforms
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip

from pytorch_msssim import ssim
import cpbd
import face_alignment

from renderer.src.utils.croper import Croper
from renderer.src.utils.preprocess import CropAndExtract

ssims,cpbds,lmd_f,lmd_m = [],[],[],[]

def extract_keypoint(images):
    detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)
    keypoints = []
    for image in images:
        image = ((image + 1) / 2.0) * 255.0
        current_kp = detector.get_landmarks_from_image(np.array(image).astype(np.uint8))[0]
        if np.mean(current_kp) == -1 and keypoints:
            keypoints.append(keypoints[-1])
        else:
            keypoints.append(current_kp[None])

    keypoints = np.concatenate(keypoints, 0)
    return keypoints

def main(args):
    pic_path = args.source_image
    save_dir = '/tmp/vq'
    os.makedirs(save_dir, exist_ok=True)
    device = args.device

    current_code_path = sys.argv[0]
    current_root_path = os.path.split(current_code_path)[0]

    os.environ['TORCH_HOME']=os.path.join(current_root_path, 'checkpoints')

    path_of_lm_croper = os.path.join(current_root_path, 'checkpoints', 'shape_predictor_68_face_landmarks.dat')
    path_of_net_recon_model = os.path.join(current_root_path, 'checkpoints', 'epoch_20.pth')
    dir_of_BFM_fitting = os.path.join(current_root_path, 'checkpoints', 'BFM_Fitting')
    croper = Croper(path_of_lm_croper)
    #init model
    preprocess_model = CropAndExtract(path_of_lm_croper, path_of_net_recon_model, dir_of_BFM_fitting, device)

    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.system(f'rm -r {first_frame_dir}')
    os.makedirs(first_frame_dir, exist_ok=True)
    first_coeff_path, crop_pic_path, frames = preprocess_model.generate(pic_path, first_frame_dir, args.preprocess, source_image_flag=True)

    full_frames = []
    clip = VideoFileClip(args.source_image)
    for frame in clip.iter_frames(fps=25):
        full_frames.append(frame)
    x_full_frames= [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  for frame in full_frames]
    x_full_frames, crop, quad = croper.crop(x_full_frames, still=True, xsize=512)
    frames_pil = [Image.fromarray(cv2.resize(frame,(256, 256))) for frame in x_full_frames]
    frames = np.stack(frames_pil, axis=0)

    gen_video = []
    clip = VideoFileClip(args.pred_path)
    for frame in clip.iter_frames(fps=25):
        frame = cv2.resize(frame, (256, 256))
        gen_video.append(frame)
    gen_video = np.stack(gen_video, axis=0)

    print(frames.shape, gen_video.shape)
    print(np.min(frames), np.max(frames))
    print(np.min(gen_video), np.max(gen_video))

    gen_video = gen_video / 255.0
    frames = frames / 255.0
    frame_num = min(frames.shape[0], gen_video.shape[0])
    frames = torch.tensor(frames[:frame_num]).permute(0,3,1,2)
    gen_video = torch.tensor(gen_video[:frame_num]).permute(0,3,1,2).to(frames.dtype)
    
    ssim_val = ssim(frames, gen_video, data_range=1, size_average=False).numpy()
    ssims.append(np.mean(ssim_val))

    for image in gen_video:
        transform = transforms.ToPILImage()
        image = transform(image)
        image = image.convert('L')
        image = np.array(image, dtype=np.float32)
        cpbds.append(cpbd.compute(image))
    
    gt_kp = extract_keypoint(frames)
    gen_kp = extract_keypoint(gen_video)

    gt_kp_mouth = gt_kp[:, 48:]
    gen_kp_mouth = gen_kp[:, 48:]
    
    dis = np.mean(np.abs(gt_kp - gen_kp))
    lmd_f.append(dis)

    dis_mouth = np.mean(np.abs(gt_kp_mouth - gen_kp_mouth))
    lmd_m.append(dis_mouth)
    print(f'SSIM: {np.mean(ssims):.4f} CPBD: {np.mean(cpbds):.4f} LMD-F: {np.mean(lmd_f):.4f} LMD-M: {np.mean(lmd_m):.4f}',flush=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--result_dir", default='results_neutral', help="path to output")
    parser.add_argument("--batch_size", type=int, default=8,  help="the batch size of facerender")
    parser.add_argument("--cpu", dest="cpu", action="store_true")
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'resize', 'full'], help="how to preprocess the images" )
    parser.add_argument("--test_path", type=str)

    parser.add_argument("--emotion", type=str)
    parser.add_argument("--level", type=str)
    
    args = parser.parse_args()
    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"
    test_path = args.test_path
    filenames = sorted(os.listdir(test_path))
    for filename in filenames:
        args.pred_path = os.path.join(test_path, filename)
        words = filename.split('.')[0].split('-')
        p_id, emotion, level, v_id = words[0], words[1], words[2], words[3], 
        args.source_image = os.path.join(f'data/MEAD/{p_id}/video/front/{emotion}/{level}/{v_id}.mp4')
        if os.path.exists(args.source_image) and os.path.exists(args.driven_audio):
            main(args)