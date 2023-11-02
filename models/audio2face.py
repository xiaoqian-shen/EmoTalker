import torch
import torch.nn as nn
from models.faceformer import Faceformer

class Audio2FaceFeature_Transformer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vqgan = self.load_vqgan(args)
        self.vqgan.eval()
        self.num_tokens = args.num_tokens

        self.audio_emb_face = nn.Sequential(nn.Linear(args.audio_dim, self.vqgan.vqvae.face_quantizer.codebook.shape[-1]))
        self.audio_emb_lip = nn.Sequential(nn.Linear(args.audio_dim, self.vqgan.vqvae.lip_quantizer.codebook.shape[-1]))

        self.faceformer = Faceformer(args, self.vqgan.vqvae.face_quantizer.codebook.shape[-1])
        self.lipformer = Faceformer(args, self.vqgan.vqvae.lip_quantizer.codebook.shape[-1])

        for p in self.vqgan.parameters():
            p.requires_grad = False

        if self.args.cls_loss:
            self.to_logits = nn.Linear(args.code_dim, args.codebook_size)

    def forward(self, x, audio_feature, sample=False):
        audio_embeddings_face = self.audio_emb_face(audio_feature)
        audio_embeddings_lip = self.audio_emb_lip(audio_feature)
        quantized_face, quantized_lip = x
        if sample:
            face_embedding = quantized_face
            lip_embedding = quantized_lip
            if x.shape[1] == 0:
                face_sos = torch.mean(audio_embeddings_face[:, :8], dim=1).unsqueeze(1)
                lip_sos = torch.mean(audio_embeddings_lip[:, :8], dim=1).unsqueeze(1)
                face_embedding = torch.cat([face_sos, face_embedding], dim=1)
                lip_embedding = torch.cat([lip_sos, lip_embedding], dim=1)
        else:
            face_embedding = quantized_face[:, :-1]
            lip_embedding = quantized_lip[:, :-1]
            face_sos = torch.mean(audio_embeddings_face[:, :8], dim=1).unsqueeze(1)
            lip_sos = torch.mean(audio_embeddings_lip[:, :8], dim=1).unsqueeze(1)
            face_embedding = torch.cat([face_sos, face_embedding], dim=1)
            lip_embedding = torch.cat([lip_sos, lip_embedding], dim=1)

        out_face = self.faceformer(face_embedding, audio_embeddings_face)
        out_lip = self.lipformer(lip_embedding, audio_embeddings_lip)
        return out_face, out_lip
    
    @staticmethod
    def load_vqgan(args):
        import models.vq as vq
        model = vq.FaceVQVAE(args)
        print('loading checkpoint from {}'.format(args.vq_ckpt))
        ckpt = torch.load(args.vq_ckpt, map_location='cpu')
        model.load_state_dict(ckpt, strict=True)
        model = model.to(args.device)
        print(model.vqvae.face_quantizer.codebook.shape)
        print(model.vqvae.lip_quantizer.codebook.shape)
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        indices = self.vqgan.encode(x)
        return indices
    
    @torch.no_grad()
    def sample(self, audio_feature):
        for k in range(torch.ceil(audio_feature.shape[1]/8)):
            if k == 0:
                face, lip = torch.empty(audio_feature.shape[0], 0, self.args.hidden_dim).to(audio_feature.device), torch.empty(audio_feature.shape[0], 0, self.args.hidden_dim).to(audio_feature.device)
            feature = self.forward((face, lip), audio_feature, sample=True)
            face = torch.cat([face, feature[:, -1].unsqueeze(1)], dim=1)
            lip = torch.cat([lip, feature[:, -1].unsqueeze(1)], dim=1)
            if k == self.num_tokens - 1:
                return face.squeeze(), lip.squeeze()
        return face.squeeze(), lip.squeeze()
