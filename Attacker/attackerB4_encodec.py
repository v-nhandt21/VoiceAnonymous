import torch, torchaudio
import sys
import soundfile as sf
import torch.nn.functional as F
from torchaudio.transforms import Resample
import torch.nn as nn

from extractor.ecapa_tdnn import ECAPA_TDNN_SMALL
from modules.retnet.retnet import RetNet

# from encodec import EncodecModel
from transformers import EncodecModel, AutoProcessor
import torchaudio

import torchaudio
import torch
import glob 
import tqdm
import sys
device = "cuda"

class CodecExtractor(torch.nn.Module):
     def __init__(self, bw):
          super().__init__()
          self.bw = bw
          self.model = EncodecModel.from_pretrained("facebook/encodec_24khz")
          self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

          self.up_resample = Resample(orig_freq=16000, new_freq=24000)
          self.down_sample = Resample(orig_freq=24000, new_freq=16000)

     def encode(self, wav):
          # [8,48000]
          wav = self.up_resample(wav)
          # [8, 72000]
          wav = wav.unsqueeze(1)
          # [8, 1, 72000]
          with torch.no_grad():
               encoded_frames = self.model.encode(wav, bandwidth=self.bw)
               # {audio_codes: [1, 8, 32, 225] , audio_scales:[]}

          return encoded_frames
     
     def decode(self, reconstruct_anon_feature_codec, anon_feature_scale):
          self.model.eval()
          # self.model.to("cpu")
          with torch.no_grad():

               if reconstruct_anon_feature_codec is None or len(reconstruct_anon_feature_codec.shape) != 4:
                    reconstruct_anon_feature_codec = torch.randint(0, 101, (1, 1, 32, 600), dtype=torch.int)

               # print("reconstruct_anon_feature_codec: ",reconstruct_anon_feature_codec.shape)
               audio_values = self.model.decode(reconstruct_anon_feature_codec, anon_feature_scale)[0]
          audio_values = self.down_sample(audio_values)

          # torchaudio.save("codec.wav", audio_values.detach().cpu()[0], sample_rate=16000)
          return audio_values



class WavlmEcapaExtractor(torch.nn.Module):
     def __init__(self):
          super().__init__()
          checkpoint = "/Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/pretrain/wavlm_large_finetune.pth"
          config_path = None 
          config_path = "/Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/pretrain/WavLM-Large.pt"
          self.extractor = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=config_path)
          state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
          self.extractor.load_state_dict(state_dict['model'], strict=False)
          self.extractor.eval()
     
     def forward(self, wav):
          with torch.no_grad():
               emb = self.extractor(wav)
          return emb

class AttackerB3(torch.nn.Module):
     def __init__(self):
          super().__init__()
          self.wavlm_ecapa_extractor = WavlmEcapaExtractor()

          self.codec_extractor = CodecExtractor(bw=24)

          self.retnet_encoder = RetNet(layers=4, hidden_dim=32, ffn_size=64, heads=8, double_v_dim=True)

     def forward(self, anon_path_waveform, orig_path_waveform):
          
          # [8, 48000]
          anon_feature = self.codec_extractor.encode(anon_path_waveform)
          anon_feature_codec, anon_feature_scale = anon_feature.audio_codes, anon_feature.audio_scales
          # [8, 32, 225]
          anon_feature_codec = anon_feature_codec[0].permute(0, 2, 1).float()
          reconstruct_anon_feature_codec = self.retnet_encoder(anon_feature_codec)
          reconstruct_anon_feature_codec = reconstruct_anon_feature_codec.permute(0, 2, 1)

          orig_feature_codec = self.codec_extractor.encode(orig_path_waveform).audio_codes[0].float() # -> [8, 32, 225]

          return reconstruct_anon_feature_codec, orig_feature_codec
     
     def inference(self, wav):

          with torch.no_grad():
               anon_feature = self.codec_extractor.encode(wav)
               anon_feature_codec, anon_feature_scale = anon_feature.audio_codes, anon_feature.audio_scales

               reconstruct_anon_feature_codec = anon_feature_codec[0].permute(0, 2, 1).float()
               reconstruct_anon_feature_codec = self.retnet_encoder(reconstruct_anon_feature_codec)
               reconstruct_anon_feature_codec = reconstruct_anon_feature_codec.permute(0, 2, 1).long()
               
               reconstruct_anon_feature_codec = reconstruct_anon_feature_codec.unsqueeze(0)

               reconstruct_audio = self.codec_extractor.decode(reconstruct_anon_feature_codec, [None]).squeeze(1).squeeze(1)
               predict_emb = self.wavlm_ecapa_extractor(reconstruct_audio)
          return predict_emb

if __name__ == "__main__":

     model = AttackerB3()

     wav, sr = sf.read("/Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/test1.wav")
     wav = torch.from_numpy(wav).unsqueeze(0).float()
     resample = Resample(orig_freq=sr, new_freq=16000)
     wav = resample(wav)
     # wav = torch.Tensor(8,48000)

     # wav:  torch.Size([2, 1, 72000])
     # audio_codes:  torch.Size([1, 2, 32, 225])

     wav2, sr = sf.read("/Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/test2.wav")
     wav2 = torch.from_numpy(wav2).unsqueeze(0).float()
     resample = Resample(orig_freq=sr, new_freq=16000)
     wav2 = resample(wav2)

     # wav2 = torch.Tensor(8,48000)
     wav = wav.to(device)
     wav2 = wav2.to(device)
     

     # Training
     model.to(device)
     output = model(wav, wav2)
     output = output[0].detach().cpu()
     print(output.shape)

     # Inference
     emb = model.inference(wav)
     print(emb.shape)
