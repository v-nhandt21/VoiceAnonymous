import torch
import soundfile as sf

from EnergyCalculator import EnergyCalculator
from PitchCalculator import Parselmouth

parsel = Parselmouth(reduction_factor=1, fs=16000)
energy_calc = EnergyCalculator(reduction_factor=1, fs=16000)

wave, sr = sf.read("/Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/test2.wav")

wave = torch.Tensor(wave)

with torch.inference_mode():
     energy = energy_calc(input_waves=wave.unsqueeze(0)).squeeze(0)
     pitch = parsel(input_waves=wave.unsqueeze(0)).squeeze(0)

     energy_average = sum(energy)/len(energy)
     pitch_average = sum(pitch)/len(pitch)

     print(energy.shape, pitch.shape)
     print(energy_average, pitch_average)