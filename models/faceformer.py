import torch
import torch.nn as nn
import math

def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                               :n - closest_power_of_2]

    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1, period).view(-1) // (period)
    bias = - torch.flip(bias, dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i + 1] = bias[-(i + 1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, period, d_model)
        repeat_num = (max_seq_len // period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Alignment Bias
def enc_dec_mask(device, T, S, k):
    mask = torch.ones(T, S)
    for i in range(T):
        start = i*k-k if i*k-k > 0 else 0
        end = i*k+k if i*k+k < S else S
        mask[i, start:end] = 0
    return (mask == 1).to(device=device)

class AudioVisualPosEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fourier_embeddings = nn.Linear(embed_dim, embed_dim)

        randn = torch.randn((2, embed_dim // 2))
        self.register_buffer("fourier_randn", randn)

    def get_fourier_embeddings(self, time_tensor):
        positions = time_tensor.unsqueeze(-1)
        positions = torch.cat((positions, positions), dim=-1)
        position_proj = (2.0 * torch.pi * positions) @ self.fourier_randn
        position_proj = torch.cat(
            [torch.sin(position_proj), torch.cos(position_proj)], dim=-1
        )
        fourier_embed = self.fourier_embeddings(position_proj)
        return fourier_embed

    def forward(self, x):
        pos_x = self.get_fourier_embeddings(x)
        return pos_x

class Faceformer(nn.Module):
    def __init__(self, args, hidden_dim, k=8):
        super(Faceformer, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.k = k
        self.biased_mask = init_biased_mask(n_head=args.n_head, max_seq_len=600, period=4)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=args.n_head,
                                                   dim_feedforward=2*self.hidden_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.num_layers)
        self.av_pos_embed = AudioVisualPosEmbedding(self.hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, token_embeddings, audio_embeddings):
        audio_timestep = torch.arange(0, audio_embeddings.shape[1]).unsqueeze(0).repeat_interleave(audio_embeddings.shape[0], dim=0)
        audio_timestep = audio_timestep.to(audio_embeddings.device)
        audio_pos_embeddings = self.av_pos_embed(audio_timestep)
        audio_embeddings = self.dropout(audio_embeddings + audio_pos_embeddings)
        
        token_timestep = torch.arange(0, token_embeddings.shape[1]).unsqueeze(0).repeat_interleave(token_embeddings.shape[0], dim=0) * 2.0
        token_timestep = token_timestep.to(token_embeddings.device)
        token_pos_embeddings = self.av_pos_embed(token_timestep)
        token_embeddings = self.dropout(token_embeddings + token_pos_embeddings)

        tgt_mask = self.biased_mask[:, :token_embeddings.shape[1], :token_embeddings.shape[1]].to(token_embeddings.device).repeat_interleave(token_embeddings.shape[0], dim=0)
        memory_mask = enc_dec_mask(token_embeddings.device, token_embeddings.shape[1], audio_embeddings.shape[1], self.k)
        out = self.transformer_decoder(token_embeddings, audio_embeddings, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return out
