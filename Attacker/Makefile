train:
	CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 nice -n 10 python train.py \
		--model_version B3_exp3b \
		--wav_scp_anon /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/filelists/wav.scp_B3 \
		--wav_scp_original /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/filelists/wav.scp_original \
		--save_dir /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/outdir/B3_exp3b \
		--epochs 30 \
		--loss_type CosineSimilarityLoss \
		--checkpoint /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/outdir/B3_exp3b/audio_model_9.pth

train2:
	CUDA_VISIBLE_DEVICES=1 taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 nice -n 10 python train.py \
		--model_version B3_exp3 \
		--wav_scp_anon /data1/nhandt23/VoicePrivacy/Attacker/resource/filelists/wav.scp_B3 \
		--wav_scp_original /data1/nhandt23/VoicePrivacy/Attacker/resource/filelists/wav.scp_original \
		--save_dir /data1/nhandt23/VoicePrivacy/Attacker/resource/outdir/B3_exp3 \
		--epochs 5 \
		--checkpoint /data1/nhandt23/VoicePrivacy/Attacker/resource/outdir/B3_exp3/audio_model_1.pth

train_triplet:
	CUDA_VISIBLE_DEVICES=0 taskset -c 11,12,13,14,15,16,17,18 nice -n 10 python train_contrastive.py \
		--model_version B3_exp4 \
		--wav_scp_anon /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/filelists/wav.scp_B3 \
		--wav_scp_original /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/filelists/wav.scp_original \
		--save_dir /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/outdir/B3_exp4 \
		--epochs 30 \
		--loss_type CosineSimilarityLoss

train_triplet_aam:
	CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,2,3,4,5,6,7,8,9,10 nice -n 10 python train_contrastive.py \
		--model_version B3_exp4 \
		--wav_scp_anon /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/filelists/wav.scp_B3 \
		--wav_scp_original /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/filelists/wav.scp_original \
		--save_dir /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/outdir/B3_exp4_aam \
		--epochs 30 \
		--loss_type CosineSimilarityLossAAM \
		--checkpoint /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/outdir/B3_exp4_aam/audio_model_2.pth

train_triplet_aam_prosody:
	CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,2,3,4,5,6,7,8,9,10 nice -n 10 python train_contrastive_prosody.py \
		--model_version B3_exp5 \
		--wav_scp_anon /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/filelists/wav.scp_B3 \
		--wav_scp_original /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/filelists/wav.scp_original \
		--save_dir /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/outdir/B3_exp5 \
		--epochs 30 \
		--loss_type CosineSimilarityLossAAM







train_ecapa_T12:
	CUDA_VISIBLE_DEVICES=0 taskset -c 11,12,13,14,15,16,17,18 nice -n 10 python train.py \
		--model_version T12_exp2 \
		--wav_scp_anon /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/filelists/wav.scp_T12 \
		--wav_scp_original /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/filelists/wav.scp_original \
		--save_dir /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/outdir/T12_exp2 \
		--epochs 30 \
		--loss_type CosineSimilarityLoss

train_mhfa_T12:
	CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,2,3,4,5,6,7,8,9,10 nice -n 10 python train.py \
		--model_version T12_exp3 \
		--wav_scp_anon /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/filelists/wav.scp_T12 \
		--wav_scp_original /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/filelists/wav.scp_original \
		--save_dir /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/outdir/T12_exp3 \
		--epochs 30 \
		--loss_type CosineSimilarityLoss



train_triplet_T12:
	CUDA_VISIBLE_DEVICES=0 taskset -c 11,12,13,14,15,16,17,18 nice -n 10 python train_contrastive.py \
		--model_version T12_exp4 \
		--wav_scp_anon /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/filelists/wav.scp_T12 \
		--wav_scp_original /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/filelists/wav.scp_original \
		--save_dir /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/outdir/T12_exp4_NoAAM \
		--epochs 30 \
		--loss_type CosineSimilarityLoss

train_triplet_aam_T12:
	CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,2,3,4,5,6,7,8,9,10 nice -n 10 python train_contrastive.py \
		--model_version T12_exp4 \
		--wav_scp_anon /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/filelists/wav.scp_T12 \
		--wav_scp_original /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/filelists/wav.scp_original \
		--save_dir /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/outdir/T12_exp4 \
		--epochs 30 \
		--loss_type CosineSimilarityLossAAM
# screen -r privacy
train_triplet_aam_prosody_T12:
	CUDA_VISIBLE_DEVICES=0 taskset -c 11,12,13,14,15,16,17,18,19,20 nice -n 10 python train_contrastive_prosody.py \
		--model_version T12_exp5 \
		--wav_scp_anon /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/filelists/wav.scp_T12 \
		--wav_scp_original /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/filelists/wav.scp_original \
		--save_dir /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/outdir/T12_exp5 \
		--epochs 30 \
		--loss_type CosineSimilarityLossAAM \
		--checkpoint /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/outdir/T12_exp5/audio_model_2.pth








# screen -r privacy2
train_codec:
	CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,2,3,4,5,6,7,8,9,10 nice -n 10 python train_encodec.py \
		--model_version B4_exp1 \
		--wav_scp_anon /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/filelists/wav.scp_B4 \
		--wav_scp_original /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/filelists/wav.scp_original \
		--save_dir /Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/resource/outdir/B4_exp1 \
		--epochs 30 \
		--loss_type MSE


