import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample
import soundfile as sf
import random

from modules.prosody.EnergyCalculator import EnergyCalculator
from modules.prosody.PitchCalculator import Parselmouth


class AudioDataset(Dataset):
     def __init__(self, wav_scp_anon, wav_scp_original, sample_rate=16000, max_length=160000):

          self.anon_file_dict = self.load_wav_scp(wav_scp_anon)
          self.orig_file_dict = self.load_wav_scp(wav_scp_original)

          self.spk2utt = self.load_spk2utt("/Data3-Processing/nhandt23_bk/VoiceAnonymous/DATA/B3/data/train-clean-360_B3/spk2utt")

          self.file_ids = list(self.anon_file_dict.keys())[:]

          self.sample_rate = sample_rate
          self.max_length = max_length

          self.speaker_list = list(self.spk2utt.keys())

          self.parsel = Parselmouth(reduction_factor=1, fs=16000)
          self.energy_calc = EnergyCalculator(reduction_factor=1, fs=16000)

     def select_triplet(self):
          # Randomly select a speaker
          speaker_id = random.choice(list(self.spk2utt.keys()))
          utterances = self.spk2utt[speaker_id]
          
          # Randomly select an anchor and positive utterance from the same speaker
          anchor_utt = random.choice(utterances)
          positive_utt = random.choice(utterances)
          
          # Ensure anchor and positive are different
          while anchor_utt == positive_utt:
               positive_utt = random.choice(utterances)
          
          # Randomly select a negative utterance from a different speaker
          negative_speaker_id = random.choice([s for s in self.spk2utt.keys() if s != speaker_id])
          negative_utt = random.choice(self.spk2utt[negative_speaker_id])

          return anchor_utt, positive_utt, negative_utt

     def select_contranstive(self):
          # Randomly select a speaker
          speaker_id = random.choice(self.speaker_list)
          positive_utt = random.choice(self.spk2utt[speaker_id])

          spk_pos_id = self.speaker_list.index(speaker_id)
          
          # Randomly select a negative utterance from a different speaker
          negative_speaker_id = random.choice([s for s in self.speaker_list if s != speaker_id])
          negative_utt = random.choice(self.spk2utt[negative_speaker_id])

          return positive_utt, negative_utt, spk_pos_id

     def load_spk2utt(self, spk2utt_file):
          """Load the spk2utt file into a dictionary."""
          spk2utt = {}
          with open(spk2utt_file, 'r') as f:
               for line in f:
                    parts = line.strip().split()
                    spk2utt[parts[0]] = parts[1:]
          return spk2utt
     
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

          return waveform[0], start

     def __getitem__(self, idx):

          # file_id = self.file_ids[idx]
          file_id_pos, file_id_neg, spk_pos_id = self.select_contranstive()

          anon_path_pos = self.anon_file_dict[file_id_pos]
          anon_path_waveform_pos, start_pos = self.load_audio(anon_path_pos)

          anon_path_neg = self.anon_file_dict[file_id_neg]
          anon_path_waveform_neg, start_neg = self.load_audio(anon_path_neg)

          orig_path = self.orig_file_dict[file_id_pos]
          orig_path_waveform, start_pos = self.load_audio(orig_path, start_pos)

          # Extract prosody from audio
          energy = self.energy_calc(input_waves=anon_path_waveform_pos.unsqueeze(0)).squeeze(0)
          pitch = self.parsel(input_waves=anon_path_waveform_pos.unsqueeze(0)).squeeze(0)
          energy_average = sum(energy)/len(energy)
          pitch_average = sum(pitch)/len(pitch)

          return anon_path_waveform_pos, anon_path_waveform_neg, orig_path_waveform, spk_pos_id, energy_average, pitch_average

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