import torch
import torch.nn.functional as F
from torch import nn

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, use_act = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual
        self.use_act = use_act

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        
        if self.use_act:
            return self.act(out)
        else:
            return out

class SimpleWrapperV2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            )
        self.args = args
        
        self.mapping1 = nn.Linear(512+64+1, 64)
        self.mapping2 = nn.Linear(13, 13)
        
        ckpt = torch.load('checkpoint/auido2exp_00300-model.pth', map_location='cpu')
        self.load_state_dict(ckpt['model'], strict=False)

        for p in self.audio_encoder.parameters():
            p.requires_grad = False

    def forward(self, x, ref):
        B = x.shape[0]
        T = x.shape[1]
        ref_reshape = ref.repeat_interleave(T, dim=0)
        x = x.reshape(B*T, 1, 80, 16)
        x = self.audio_encoder(x).view(x.size(0), -1)
        ratio = torch.zeros(x.shape[0], 1).to(self.args.device)
        y = self.mapping1(torch.cat([x, ref_reshape, ratio], dim=1))
        lip_indices = torch.tensor([0,1,2,3,4,5,7,9,10,11,12,13,14]).to(y.device)
        y_finetune = self.mapping2(y[:, lip_indices])
        y = self.args.lip_mix * y_finetune + (1 - self.args.lip_mix) * y[:, lip_indices]
        out = y.reshape(B, T, -1)
        return out

    def test_freeze_linear(self, x, ref, ratio):
        x = self.audio_encoder(x.float()).view(x.size(0), -1)
        ref_reshape = ref.reshape(x.size(0), -1)
        ratio = ratio.to(self.args.device)
        y = self.mapping1(torch.cat([x, ref_reshape, ratio], dim=1))
        lip_indices = torch.tensor([0,1,2,3,4,5,7,9,10,11,12,13,14]).to(y.device)
        y_finetune = self.mapping2(y[:, lip_indices])
        y = self.args.lip_mix * y_finetune + (1 - self.args.lip_mix) * y[:, lip_indices]
        out = y.reshape(ref.shape[0], 13)
        return out