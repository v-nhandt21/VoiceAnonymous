import torch, torchaudio
import torch.nn as nn
from torchaudio.transforms import Resample

def stereo_to_mono_convertor(signal):
     if signal.shape[0] > 1:
          signal = torch.mean(signal, dim=0, keepdim=True)
     return signal

def load_audio(audio_file):
     audio,sr = torchaudio.load(audio_file)
     resample = Resample(orig_freq=sr, new_freq=16000)
     audio = resample(audio)
     wav = stereo_to_mono_convertor(audio)
     return wav 

def load_model(model_path, freeze=True):
     checkpoint = torch.load(model_path)
     cfg = WavLMConfig(checkpoint['cfg'])
     model = WavLM(cfg)
     model.load_state_dict(checkpoint['model'])

     if freeze:
          for param in model.parameters():
               param.requires_grad = False
          model.eval()
     return model

def get_feature_last_layer(model, wav):
     wav = torch.nn.functional.layer_norm(wav , wav.shape)
     rep = model.extract_features(wav)[0]
     return rep

def get_feature_average_layer(model, wav):
     wav = torch.nn.functional.layer_norm(wav , wav.shape)
     # For package triton
     # wav = torch.nn.functional.layer_norm(wav[:,:48000] , [1,48000])
     
     rep, layer_results = model.extract_features(wav, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
     layer_reps = [x.transpose(0, 1) for x, _ in layer_results]

     average_tensor = torch.stack(layer_reps).sum(dim=0) / len(layer_reps)

     average_tensor = torch.nn.functional.pad(average_tensor, (0, 0, 0, 1))
     average_tensor[:,-1,:] = average_tensor[:,-2,:]

     return average_tensor

def get_full_layer_feature(model, wav):
     wav = torch.nn.functional.layer_norm(wav , wav.shape)
     # For package triton
     # wav = torch.nn.functional.layer_norm(wav[:,:48000] , [1,48000])
     
     rep, layer_results = model.extract_features(wav, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
     layer_reps = [x.transpose(0, 1) for x, _ in layer_results]

     x = torch.stack(layer_reps).transpose(0,-1).transpose(0,1)

     return x

if __name__ == "__main__":
     from wavlm import WavLM, WavLMConfig
     
     model = load_model('/home/nhandt23/Desktop/VoiceDiarization/diarizationservice/L7_VoiceFilter/wavlm/WavLM-Large.pt')
     wav = load_audio('/home/nhandt23/Desktop/VoiceDiarization/diarizationservice/L7_VoiceFilter/simulate_test.wav')
     wav = wav.to("cpu")

     feature = get_feature_average_layer(model, wav)

     print(feature.shape)

     print("-------------------------")
     wav = load_audio('/home/nhandt23/Desktop/VoiceDiarization/diarizationservice/L7_VoiceFilter/DataSimulationOnthefly/simulate_test.wav')
     wav = wav.to("cpu")

     feature = get_feature_average_layer(model, wav)

     print(feature.shape)

     print(feature[0][149])
else:
     from .wavlm import WavLM, WavLMConfig