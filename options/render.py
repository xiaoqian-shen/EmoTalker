import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description='Transformer',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader
    parser.add_argument('--clip_len', default=96, type=int)
    parser.add_argument('--num_workers', default=5, type=int, help='batch size')
    parser.add_argument('--path', type=str, default='data/mead-3dmm', help='batch size')

    ## transformer
    parser.add_argument('--num_layers', default=6, type=int, help='n_layers')
    parser.add_argument('--n_head', default=4, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int, help='hidden_dim')
    parser.add_argument('--num_tokens', default=24, type=int, help='num_tokens') # input_len / down_t**2
    parser.add_argument('--sample_rate', default=16000, type=int, help='sample_rate')
    parser.add_argument('--audio_dim', default=768, type=int, help='audio hubert feature dim')
    parser.add_argument('--update_audio', action='store_true', default=False)
    parser.add_argument('--freeze', type=bool, default=False)
    parser.add_argument('--joint_pos', type=bool, default=False)

    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--accum-grad', type=int, default=1, help='Number for gradient accumulation.')

    parser.add_argument('--vq_ckpt', type=str, default='ckpt/vq_face/vq_face-2023_04_25_16_17_14/vq.pt', help='load first stage model')
    parser.add_argument('--trans_ckpt', type=str, default=None, help='load first stage model')
    parser.add_argument('--start_from_epoch', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_ckpt', type=int, default=10)

    parser.add_argument('--resume_vq', action='store_true', default=False)
    parser.add_argument('--resume_trans', action='store_true', default=False)

    ## vqvae arch
    parser.add_argument("--input_dim", type=int, default=70, help="embedding dimension")
    parser.add_argument("--input_dim_face", type=int, default=57, help="embedding dimension")
    parser.add_argument("--input_dim_lip", type=int, default=13, help="embedding dimension")
    parser.add_argument("--face_code_dim", type=int, default=256, help="embedding dimension")
    parser.add_argument("--code_dim", type=int, default=256, help="embedding dimension")
    parser.add_argument("--face_codebook_size", type=int, default=1024, help="nb of embedding")
    parser.add_argument("--codebook_size", type=int, default=1024, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down_t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride_t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=4, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument('--vq-act', type=str, default='relu', choices=['relu', 'silu', 'gelu'],
                        help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')
    parser.add_argument("--cls_num", type=int, default=8, help="emotional classes num")

    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices=['ema', 'orig', 'ema_reset', 'reset'],
                        help="eps for optimal transport")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')

    parser.add_argument("--lip_mix", type=float, default=0.2)
    parser.add_argument("--contra", type=float, default=0.8)

    # render
    parser.add_argument("--driven_audio", default='./examples/driven_audio/bus_chinese.wav', help="path to driven audio")
    parser.add_argument("--source_image", default='./examples/source_image/full_body_2.png', help="path to source image")
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default='checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./result_neutral', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2,  help="the batch size of facerender")
    parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    parser.add_argument('--enhancer',  type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]")
    parser.add_argument('--background_enhancer',  type=str, default=None, help="background enhancer, [realesrgan]")
    parser.add_argument("--cpu", dest="cpu", action="store_true") 
    parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks") 
    parser.add_argument("--still", action="store_true", help="can crop back to the original videos for the full body aniamtion") 
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'resize', 'full'], help="how to preprocess the images" ) 

    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='useless')
    parser.add_argument('--init_path', type=str, default=None, help='Useless')
    parser.add_argument('--use_last_fc',default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # default renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    parser.add_argument('--save_name', type=str, required=True)
    parser.add_argument('--neutral', default=False, action="store_true")
    parser.add_argument('--select', default=False, action="store_true")
    parser.add_argument('--random', default=False, action="store_true")
    parser.add_argument('--prefix', default=None, type=str)
    parser.add_argument('--use_DAIN', default=False, action="store_true")

    parser.add_argument("--enhancer_region", type=str, default='lip', help="enhaner region:[none,lip,face] \
                                                                      none:do not enhance; \
                                                                      lip:only enhance lip region \
                                                                      face: enhance (skin nose eye brow lip) region")

    return parser.parse_args()