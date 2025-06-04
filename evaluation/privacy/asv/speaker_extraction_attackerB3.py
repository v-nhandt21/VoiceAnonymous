import torch, torchaudio
import sys

import soundfile as sf

import torch.nn.functional as F
from torchaudio.transforms import Resample

device = "cuda"

class SpeakerExtractionCustom:
     def __init__(self):
          self.embedding_collection = {}
          self.spk2utt = {}
          self.wav_scp_collection = {}

          sys.path.append("/Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker")

          model_version = "B3_exp5"
          checkpoint = "/Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/outdir/B3_exp5/audio_model_8.pth"
          
          # model_version = "B3_exp4"
          # checkpoint = "/Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/outdir/B3_exp4_aam/audio_model_16.pth"

          model_version = "B4_exp1"
          checkpoint = "/Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/outdir/B4_exp1/audio_model_10.pth"



          model_version = "T12_exp5"
          checkpoint = "/Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/outdir/T12_exp5/audio_model_5.pth"

          model_version = "T12_exp4"
          checkpoint = "/Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/outdir/T12_exp4_NoAAM/audio_model_1.pth"

          model_version = "T12_exp4"
          checkpoint = "/Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/outdir/T12_exp4/audio_model_6.pth"

          model_version = "T12_exp3"
          checkpoint = "/Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/outdir/T12_exp3/audio_model_9.pth"

          model_version = "T12_exp2"
          checkpoint = "/Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/outdir/T12_exp2/audio_model_9.pth"

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
          if model_version == "T12_exp4":
               from attackerB3_exp4_triplet import AttackerB3
          if model_version == "B4_exp1":
               from attackerB4_encodec import AttackerB3
          if model_version == "T12_exp3":
               from attackerB3_exp3_mhfa import AttackerB3
          if model_version == "T12_exp2":
               from attackerB3_exp2_ecapa import AttackerB3
               
          self.model_version = model_version
          self.model = AttackerB3()


          state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
          self.model.load_state_dict(state_dict, strict=False)
          self.model.to(device)
          self.model.eval()

     def get_embedding(self, dataset_path, ids, emb_level):

          dataset_path = str(dataset_path)
          
          if dataset_path in self.embedding_collection:
               if ids in self.embedding_collection[dataset_path]:
                    return self.embedding_collection[dataset_path][ids]

          if dataset_path not in self.embedding_collection:
               self.embedding_collection[dataset_path] = {}

          if dataset_path not in self.wav_scp_collection:
               self.wav_scp_collection[dataset_path] = {}
               wav_scp_file = dataset_path + "/wav.scp"
               wav_scps = open(wav_scp_file, "r", encoding="utf-8").read().splitlines()
               for wav_scp in wav_scps:
                    wav_scp = wav_scp.split(" ")
                    self.wav_scp_collection[dataset_path][wav_scp[0]] = wav_scp[1]

          if emb_level == "utt":
               utt = self.wav_scp_collection[dataset_path][ids]
               # print(emb_level, utt)
               # emb = torch.rand(256)

               try:
                    # print(utt)
                    # emb = torch.rand(256) 
                    emb = self.inference(utt)
               except:
                    emb = torch.rand(256)
          elif emb_level == "spk":

               if ids not in self.spk2utt:
                    spk2utt_file = dataset_path + "/spk2utt"
                    spks = open(spk2utt_file, "r", encoding="utf-8").read().splitlines()
                    for spk in spks:
                         spk = spk.split(" ")
                         self.spk2utt[spk[0]] = spk[1:]

               utts = []
               for i in self.spk2utt[ids]:          
                    utt = self.wav_scp_collection[dataset_path][i]
                    utts.append(utt)

               # print(emb_level, utts)
               # emb = torch.rand(256)
               emb = []
               for utt in utts:
                    e = self.inference(utt)
                    emb.append(e)
               emb = torch.mean(torch.stack(emb), dim=0)

          # self.embedding_collection[dataset_path][ids] = emb 

          return emb

     def inference(self, utt):
          # https://huggingface.co/microsoft/wavlm-base-plus-sv
          # https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification
          wav, sr = sf.read(utt)

          wav = torch.from_numpy(wav).unsqueeze(0).float()
          resample = Resample(orig_freq=sr, new_freq=16000)
          wav = resample(wav)

          wav = wav.to(device)

          with torch.no_grad():

               if self.model_version == "B3_exp5" or self.model_version == "T12_exp5":
                    emb, _, _, _ = self.model(wav)
               elif self.model_version == "B4_exp1":
                    emb = self.model.inference(wav)
               else:
                    emb, _ = self.model(wav)

               emb = emb.detach().cpu()
          # sim = F.cosine_similarity(emb1, emb2)

          return emb[0]