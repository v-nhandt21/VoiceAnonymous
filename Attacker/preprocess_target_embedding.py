import torch
from extractor.ecapa_tdnn import ECAPA_TDNN_SMALL
import soundfile as sf
import numpy as np

device = "cuda"

class WavlmEcapaExtractor(torch.nn.Module):
     def __init__(self):
          super().__init__()
          checkpoint = "/data1/nhandt23/UniSpeech/downstreams/speaker_verification/wavlm_large_finetune.pth"
          config_path = None 
          config_path = "/data1/nhandt23/VoicePrivacy/Attacker/wavlm/WavLM-Large.pt"
          self.extractor = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=config_path)
          state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
          self.extractor.load_state_dict(state_dict['model'], strict=False)
          self.extractor.eval()
     
     def forward(self, wav):
          with torch.no_grad():
               emb = self.extractor(wav)
          return emb

def load_wav_scp(wav_scp_file):
     wav_dict = {}
     with open(wav_scp_file, 'r') as f:
          for line in f:
               parts = line.strip().split()  # Split on the first whitespace

               key, value = parts[0], parts[-1]
               wav_dict[key] = value
     return wav_dict

if __name__ == "__main__":

     file_dict = load_wav_scp("/Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/filelists/wav.scp_original")

     wavlm_ecapa_extractor = WavlmEcapaExtractor()

     for k, path in file_dict.items():

          waveform, sample_rate = sf.read(path)
          waveform = torch.from_numpy(waveform).unsqueeze(0).float()
          
          emb = wavlm_ecapa_extractor(waveform).detach().cpu().numpy()
          np.save('/Data3-Processing/nhandt23_bk/VoiceAnonymous/DATA/raw_wavlm_ecapa_emb/' + k + '.npy', emb)