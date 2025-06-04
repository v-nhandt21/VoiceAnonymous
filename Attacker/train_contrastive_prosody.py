import torch
import torch.nn as nn
import torch.optim as optim
from dataloader_contrastive_prosody import get_dataloader
import tqdm, os
from torch.utils.tensorboard import SummaryWriter
from loss.cosin_loss import CosineSimilarityLoss
from loss.aamsoftmax import AAMSoftmaxLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, wav_scp_anon, wav_scp_original, save_dir, epochs, cur_epoch):

     batch_size = 12

     use_prosody_loss = True
     if use_prosody_loss:
          prosody_energy_loss = nn.MSELoss()
          prosody_pitch_loss = nn.MSELoss()

     if loss_type == "CosineSimilarityLossAAM":
          use_aamsoftmax = True
     else:
          use_aamsoftmax = False 

     if use_aamsoftmax:
          aam_softmax = AAMSoftmaxLoss(256, speaker_num=1000)
     
     if loss_type == "MSE":
          criterion = nn.MSELoss()
     elif loss_type == "CosineSimilarityLoss":
          criterion = CosineSimilarityLoss()
     elif loss_type == "CosineSimilarityLossAAM":
          criterion = CosineSimilarityLoss()
     else:
          criterion = nn.MSELoss()

     optimizer = optim.Adam(model.parameters(), lr=0.001)

     os.makedirs(save_dir, exist_ok=True)

     dataloader = get_dataloader(wav_scp_anon, wav_scp_original, batch_size=batch_size)

     writer = SummaryWriter(log_dir=save_dir + "/logdir")

     iteration = 0

     # Accumulation - because batch-size is small
     accumulation_iteration_steps = 3
     total_loss = 0.0
     num_samples = 0

     for epoch in tqdm.tqdm(range(cur_epoch, epochs, 1), initial=cur_epoch, total=epochs):

          print("Epoch: ", epoch)
          
          running_loss = 0.0
          for batch_data in tqdm.tqdm(dataloader):
               anon_path_waveform_pos, anon_path_waveform_neg, orig_path_waveform, spk_pos_id, energy_average, pitch_average = batch_data

               anon_path_waveform_pos, anon_path_waveform_neg, orig_path_waveform = anon_path_waveform_pos.to(device), anon_path_waveform_neg.to(device), orig_path_waveform.to(device)

               energy_average, pitch_average = energy_average.to(device), pitch_average.to(device)
               # optimizer.zero_grad()

               predict_orig_emb_pos, orig_emb, energy_predict, pitch_predict = model(anon_path_waveform_pos, orig_path_waveform)
               predict_orig_emb_neg, orig_emb, energy_predict, pitch_predict = model(anon_path_waveform_neg, orig_path_waveform)

               loss = criterion(predict_orig_emb_pos, orig_emb) - criterion(predict_orig_emb_neg, orig_emb) + 1

               if use_aamsoftmax:
                    loss += aam_softmax(predict_orig_emb_pos, spk_pos_id)

               if use_prosody_loss:
                    loss += prosody_energy_loss(energy_predict, energy_average)
                    loss += prosody_pitch_loss(pitch_predict, pitch_average)

               # loss.backward()
               # optimizer.step()
               # optimizer.zero_grad()

               total_loss += loss 
               num_samples += len(spk_pos_id)
               if iteration % accumulation_iteration_steps:
                    average_loss = total_loss / num_samples

                    average_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    total_loss = 0.0
                    num_samples = 0

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
     if model_version == "B3_exp4":
          from attackerB3_exp4_triplet import AttackerB3
     if model_version == "B3_exp5":
          from attackerB3_exp5_grl_prosody import AttackerB3
     if model_version == "T12_exp5":
          from attackerT12_exp5_grl_prosody import AttackerB3
     
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

     
