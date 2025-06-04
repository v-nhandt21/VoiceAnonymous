import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_dataloader
import tqdm, os
from torch.utils.tensorboard import SummaryWriter
from loss.cosin_loss import CosineSimilarityLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, wav_scp_anon, wav_scp_original, save_dir, epochs, cur_epoch):

     batch_size = 24
     
     if loss_type == "MSE":
          criterion = nn.MSELoss()
     elif loss_type == "CosineSimilarityLoss":
          criterion = CosineSimilarityLoss()
     else:
          criterion = nn.MSELoss()

     optimizer = optim.Adam(model.parameters(), lr=0.001)

     os.makedirs(save_dir, exist_ok=True)

     dataloader = get_dataloader(wav_scp_anon, wav_scp_original, batch_size=batch_size)

     writer = SummaryWriter(log_dir=save_dir + "/logdir")

     iteration = 0
     for epoch in tqdm.tqdm(range(cur_epoch, epochs, 1), initial=cur_epoch, total=epochs):

          print("Epoch: ", epoch)
          
          running_loss = 0.0
          for batch_data in tqdm.tqdm(dataloader):
               anon_path_waveform, orig_path_waveform = batch_data
               anon_path_waveform, orig_path_waveform = anon_path_waveform.to(device), orig_path_waveform.to(device)

               optimizer.zero_grad()
               predict_orig_emb, orig_emb = model(anon_path_waveform, orig_path_waveform)
               loss = criterion(predict_orig_emb, orig_emb)
               loss.backward()
               optimizer.step()

               running_loss += loss.item()

               iteration += 1
               if iteration%10 == 0:
                    writer.add_scalar('loss', loss/batch_size, iteration)
          
          print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")

          torch.save(model.state_dict(), save_dir + "/audio_model_" + str(epoch) + ".pth")
          print("Model saved as audio_model.pth")

def get_parser():
    parser = argparse.ArgumentParser(description="Process audio files and model configurations.")
    
    parser.add_argument('--model_version', type=str, required=True, help='Version of the model to use.')
    parser.add_argument('--wav_scp_anon', type=str, required=True, help='Path to the anonymized WAV SCP file.')
    parser.add_argument('--wav_scp_original', type=str, required=True, help='Path to the original WAV SCP file.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the outputs.')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs.')
    parser.add_argument('--checkpoint', default="", type=str, required=False, help='Checkpoint.')
    parser.add_argument('--loss_type', default="", type=str, required=True, help='Loss.')
    
    return parser

if __name__ == "__main__":
     import glob, argparse

     parser = get_parser()
     args = parser.parse_args()

     model_version = args.model_version
     wav_scp_anon = args.wav_scp_anon
     wav_scp_original = args.wav_scp_original
     save_dir = args.save_dir
     epochs = args.epochs
     checkpoint = args.checkpoint
     loss_type = args.loss_type

     if model_version == "B3_exp1":
          from attackerB3_exp1_linear import AttackerB3
     if model_version == "B3_exp2":
          from attackerB3_exp2_ecapa import AttackerB3
     if model_version == "B3_exp3":
          from attackerB3_exp3_mhfa import AttackerB3
     if model_version == "B3_exp3b":
          from attackerB3_exp3_mhfa import AttackerB3
     if model_version == "T12_exp3":
          from attackerB3_exp3_mhfa import AttackerB3
     if model_version == "T12_exp2":
          from attackerB3_exp2_ecapa import AttackerB3

     model = AttackerB3().to(device)

     if checkpoint != "":
          model.load_state_dict(torch.load(checkpoint))
          cur_epoch = int(checkpoint.split("_")[-1].split(".")[0]) + 1
     else:
          cur_epoch = 0

     train(model, wav_scp_anon, wav_scp_original, save_dir, epochs, cur_epoch)

     # from attackerB3 import AttackerB3
     # model = AttackerB3().to(device)
     # wav_scp_anon = "/data1/nhandt23/VoicePrivacy/Attacker/resource/filelists/wav.scp_B3"
     # wav_scp_original = "/data1/nhandt23/VoicePrivacy/Attacker/resource/filelists/wav.scp_original"
     # save_dir = "/data1/nhandt23/VoicePrivacy/Attacker/resource/outdir/B3_exp1"
     # epochs = 5

     # from attackerB3_exp2_ecapa import AttackerB3
     # model = AttackerB3().to(device)
     # wav_scp_anon = "/data1/nhandt23/VoicePrivacy/Attacker/resource/filelists/wav.scp_B3"
     # wav_scp_original = "/data1/nhandt23/VoicePrivacy/Attacker/resource/filelists/wav.scp_original"
     # save_dir = "/data1/nhandt23/VoicePrivacy/Attacker/resource/outdir/B3_exp2"
     # epochs = 5

     # from attackerB3_exp3_mhfa import AttackerB3
     # model = AttackerB3().to(device)
     # wav_scp_anon = "/data1/nhandt23/VoicePrivacy/Attacker/resource/filelists/wav.scp_B3"
     # wav_scp_original = "/data1/nhandt23/VoicePrivacy/Attacker/resource/filelists/wav.scp_original"
     # save_dir = "/data1/nhandt23/VoicePrivacy/Attacker/resource/outdir/B3_exp3"
     # epochs = 5

     
