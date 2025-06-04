import torch
import torchaudio
from attackerB3 import AttackerB3
from dataloader import AudioDataset

def load_model(model_path="audio_model.pth"):
     model = AttackerB3()
     model.load_state_dict(torch.load(model_path))
     model.eval()
     return model

def run_inference(audio_path):
     model = load_model()
     dataset = AudioDataset([audio_path], [torch.zeros(256)])  # Dummy target for inference
     waveform, _ = dataset[0]
     
     with torch.no_grad():
          output = model(waveform.unsqueeze(0))  # Add batch dimension
     return output[0]

if __name__ == "__main__":
     audio_path = "path/to/new_audio.wav"
     output = run_inference(audio_path)
     print("Inference output:", output)
