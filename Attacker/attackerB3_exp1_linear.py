import torch, torchaudio
import sys
import soundfile as sf
import torch.nn.functional as F
from torchaudio.transforms import Resample
import torch.nn as nn

from extractor.ecapa_tdnn import ECAPA_TDNN_SMALL

device = "cuda"

class WavlmEcapaExtractor(torch.nn.Module):
     def __init__(self):
          super().__init__()
          checkpoint = "/data1/nhandt23/UniSpeech/downstreams/speaker_verification/wavlm_large_finetune.pth"
          config_path = None 
          config_path = "/data1/nhandt23/VoicePrivacy/Attacker/modules/wavlm/WavLM-Large.pt"
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
          self.fc1 = nn.Linear(256, 512)
          self.fc2 = nn.Linear(512, 256)

     def forward(self, anon_path_waveform, orig_path_waveform=None):

          anon_emb = self.wavlm_ecapa_extractor(anon_path_waveform)

          if orig_path_waveform is not None:
               orig_emb = self.wavlm_ecapa_extractor(orig_path_waveform)
          else:
               orig_emb = None

          x = self.fc1(anon_emb)
          predict_orig_emb = self.fc2(x)

          return predict_orig_emb, orig_emb

if __name__ == "__main__":

     model = AttackerB3()

     wav, sr = sf.read("/data1/nhandt23/VoicePrivacy/Attacker/test1.wav")
     wav = torch.from_numpy(wav).unsqueeze(0).float()
     resample = Resample(orig_freq=sr, new_freq=16000)
     wav = resample(wav)

     wav2, sr = sf.read("/data1/nhandt23/VoicePrivacy/Attacker/test2.wav")
     wav2 = torch.from_numpy(wav2).unsqueeze(0).float()
     resample = Resample(orig_freq=sr, new_freq=16000)
     wav2 = resample(wav2)

     wav = wav.to(device)
     wav2 = wav2.to(device)
     model.to(device)

     output = model(wav, wav2)

     output = output[0].detach().cpu()

     print(output.shape)