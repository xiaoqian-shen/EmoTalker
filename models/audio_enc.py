import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification

class AudioEncoder(nn.Module):
    def __init__(self, args):
        super(AudioEncoder, self).__init__()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-er")
        self.encoder = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-er")
        self.sample_rate = args.sample_rate

    def forward(self, audio):
        encoded = self.processor(audio.tolist(), padding=True, return_tensors="pt", sampling_rate=self.sample_rate)
        attn_mask = encoded.attention_mask.to(audio.device)
        self.encoder.eval()
        with torch.no_grad():
            output = self.encoder(encoded.input_values.to(audio.device), attention_mask=attn_mask)
            features = output.hidden_states[-1].detach()
        # features = features.masked_fill(attn_mask[..., None], 0.)
        return features