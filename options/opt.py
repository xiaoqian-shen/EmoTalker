import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description='Transformer',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader
    parser.add_argument('--clip_len', default=96, type=int)
    parser.add_argument('--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('--batch_size_cond', default=256, type=int, help='batch size')
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

    ## optimization
    parser.add_argument('--total-iter', default=20000, type=int, help='number of total iterations to run')
    parser.add_argument('--epochs', default=10000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=300, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=1e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[1000, 50000], nargs="+", type=int,
                        help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    parser.add_argument('--lr_scheduler', default=[6000], nargs="+", type=int, help="learning rate schedule (iterations)")

    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.2, help="hyper-parameter for the commitment loss")
    parser.add_argument('--recons-loss', type=str, default='l2', help='reconstruction loss')
    parser.add_argument("--emotion", type=float, default=0.2, help="emotion loss")
    parser.add_argument("--contra", type=float, default=0.8, help="emotion loss")
    parser.add_argument("--lip_mix", type=float, default=0.2, help="emotion loss")

    ## output directory
    parser.add_argument('--out-dir', type=str, default='output_vqfinal/', help='output directory')
    parser.add_argument('--results-dir', type=str, default='visual_results/', help='output directory')
    parser.add_argument('--visual-name', type=str, default='baseline', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug',
                        help='name of the experiment, will create a file inside out-dir')
    parser.add_argument('--pkeep', type=float, default=1.0, help='keep rate for gpt training')
    ## other
    parser.add_argument('--save_iter', default=1000, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=1000, type=int, help='evaluation frequency')
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')

    return parser.parse_args()