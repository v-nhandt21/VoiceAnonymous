import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample
import soundfile as sf
import random

class AudioDataset(Dataset):
     def __init__(self, wav_scp_anon, wav_scp_original, sample_rate=16000, max_length=160000):

          self.anon_file_dict = self.load_wav_scp(wav_scp_anon)
          self.orig_file_dict = self.load_wav_scp(wav_scp_original)

          self.file_ids = list(self.anon_file_dict.keys())[:]

          self.sample_rate = sample_rate
          self.max_length = max_length

     def load_wav_scp(self, wav_scp_file):
          wav_dict = {}
          with open(wav_scp_file, 'r') as f:
               for line in f:
                    parts = line.strip().split()  # Split on the first whitespace

                    key, value = parts[0], parts[-1]
                    wav_dict[key] = value
          return wav_dict

     def __len__(self):
          return len(self.file_ids)

     def load_audio(self, path, start=None):
          waveform, sample_rate = sf.read(path)
          waveform = torch.from_numpy(waveform).unsqueeze(0).float()

          # waveform, sample_rate = torchaudio.load(path)

          # waveform, sample_rate = torchaudio.backend.soundfile_backend.load(path)
                                    
          if sample_rate != self.sample_rate:
               print("Resample rate for: ", path)
               resample = Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
               waveform = self.resample(waveform)
          
          padding = self.max_length - waveform.size(1)
          if padding > 0:
               waveform = torch.nn.functional.pad(waveform, (0, padding))
          else:
               start = random.randint(0, waveform.size(1) - self.max_length) if start is None else start
               waveform = waveform[:, start:start + self.max_length]

          padding = self.max_length - waveform.size(1)
          if padding > 0:
               waveform = torch.nn.functional.pad(waveform, (0, padding))

          return waveform[0], start

     def __getitem__(self, idx):

          file_id = self.file_ids[idx]

          anon_path = self.anon_file_dict[file_id]
          anon_path_waveform, start = self.load_audio(anon_path)

          orig_path = self.orig_file_dict[file_id]
          orig_path_waveform, start = self.load_audio(orig_path, start)

          return anon_path_waveform, orig_path_waveform

def get_dataloader(wav_scp_anon, wav_scp_original, batch_size=32):
     dataset = AudioDataset(wav_scp_anon, wav_scp_original)
     return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":

     file_paths = ["test1.wav", "test2.wav"]

     wav_scp_anon = "/data1/nhandt23/VoicePrivacy/Attacker/resource/filelists/wav.scp_B3"
     wav_scp_original = "/data1/nhandt23/VoicePrivacy/Attacker/resource/filelists/wav.scp_original"

     dataloader = get_dataloader(wav_scp_anon, wav_scp_original, batch_size=2)
     for waveforms, targets in dataloader:
          print("waveforms: ", waveforms.shape)
          print("targets: ", targets.shape)