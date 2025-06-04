# Voice Attacker

**Paper:** *Leveraging Multi-Head Factorized Attentive Reconstructor and Gradient Reversal for Random Prosody Anonymization - Voice Privacy Challenge*

---

## 🔧 Setup

Ensure you have `conda` installed. Then, run the following command to install the necessary dependency:

```bash
conda install 'ffmpeg<5'
```

---

## 🔐 Data and Model Download

> **Download Password:** `getdata`

Run the following script to download the required data and pretrained models:

```bash
bash 01_download_data_model.sh
```

---

# 🎯 Voice Anonymization Attack Model Training

This section outlines training commands for various attack models targeting voice anonymization systems. The models are trained using different architectures and loss functions, evaluated on both original and anonymized speech.

---

## 🏗️ Training Commands

### 🔹 ECAPA Model Training

#### T12 Experiment 2

```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 11-18 nice -n 10 python train.py \
  --model_version T12_exp2 \
  --wav_scp_anon VoiceAnonymous/Attacker/resource/filelists/wav.scp_T12 \
  --wav_scp_original VoiceAnonymous/Attacker/resource/filelists/wav.scp_original \
  --save_dir VoiceAnonymous/Attacker/resource/outdir/T12_exp2 \
  --epochs 30 \
  --loss_type CosineSimilarityLoss
```

---

### 🔹 MHFA Model Training

#### T12 Experiment 3

```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0-10 nice -n 10 python train.py \
  --model_version T12_exp3 \
  --wav_scp_anon VoiceAnonymous/Attacker/resource/filelists/wav.scp_T12 \
  --wav_scp_original VoiceAnonymous/Attacker/resource/filelists/wav.scp_original \
  --save_dir VoiceAnonymous/Attacker/resource/outdir/T12_exp3 \
  --epochs 30 \
  --loss_type CosineSimilarityLoss
```

---

### 🔹 Triplet Model Training (No AAM)

#### T12 Experiment 4

```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 11-18 nice -n 10 python train_contrastive.py \
  --model_version T12_exp4 \
  --wav_scp_anon VoiceAnonymous/Attacker/resource/filelists/wav.scp_T12 \
  --wav_scp_original VoiceAnonymous/Attacker/resource/filelists/wav.scp_original \
  --save_dir VoiceAnonymous/Attacker/resource/outdir/T12_exp4_NoAAM \
  --epochs 30 \
  --loss_type CosineSimilarityLoss
```

---

### 🔹 Triplet Model Training (With AAM)

```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0-10 nice -n 10 python train_contrastive.py \
  --model_version T12_exp4 \
  --wav_scp_anon VoiceAnonymous/Attacker/resource/filelists/wav.scp_T12 \
  --wav_scp_original VoiceAnonymous/Attacker/resource/filelists/wav.scp_original \
  --save_dir VoiceAnonymous/Attacker/resource/outdir/T12_exp4 \
  --epochs 30 \
  --loss_type CosineSimilarityLossAAM
```

---

### 🔹 Triplet + Prosody Model Training (With AAM)

```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 11-20 nice -n 10 python train_contrastive_prosody.py \
  --model_version T12_exp5 \
  --wav_scp_anon VoiceAnonymous/Attacker/resource/filelists/wav.scp_T12 \
  --wav_scp_original VoiceAnonymous/Attacker/resource/filelists/wav.scp_original \
  --save_dir VoiceAnonymous/Attacker/resource/outdir/T12_exp5 \
  --epochs 30 \
  --loss_type CosineSimilarityLossAAM \
  --checkpoint VoiceAnonymous/Attacker/resource/outdir/T12_exp5/audio_model_2.pth
```

---

### 🔹 Codec Model Training

```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0-10 nice -n 10 python train_encodec.py \
  --model_version B4_exp1 \
  --wav_scp_anon VoiceAnonymous/Attacker/resource/filelists/wav.scp_B4 \
  --wav_scp_original VoiceAnonymous/Attacker/resource/filelists/wav.scp_original \
  --save_dir VoiceAnonymous/Attacker/resource/outdir/B4_exp1 \
  --epochs 30 \
  --loss_type MSE
```

---

# 🚀 Attack Scenarios

### 🔹 Baseline Attack

#### Pre-Attack Inference

```bash
python inference.py --config configs/eval_pre_attacker.yaml --overwrite "{\"anon_data_suffix\": \"_B3\"}" --force_compute True
```

#### Post-Attack Evaluation

```bash
python run_evaluation.py --config configs/eval_post_attacker.yaml --overwrite "{\"anon_data_suffix\": \"_B3\"}" --force_compute True
```

---

## 🧪 Additional Inference Runs

### 🔸 Baseline Inference (No Evaluation)

```bash
python inference.py --config configs/eval_pre_attacker.yaml --overwrite "{\"anon_data_suffix\": \"_B3\"}"
```

### 🔸 SpeechWorld Inference Variants

#### ▪️ B3 Configuration (GPU 1)

```bash
CUDA_VISIBLE_DEVICES=1 python inference.py --config configs/eval_speechworld_B3.yaml --overwrite "{\"anon_data_suffix\": \"_B3\"}" --force_compute True
```

#### ▪️ B4 Configuration (GPU 0, Cores 0–10)

```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0-10 nice -n 10 python inference.py --config configs/eval_speechworld_B4.yaml --overwrite "{\"anon_data_suffix\": \"_B4\"}" --force_compute True
```

#### ▪️ T12-5 Configuration (GPU 0, Cores 10–19)

```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 10-19 nice -n 10 python inference.py --config configs/eval_speechworld_T12-5.yaml --overwrite "{\"anon_data_suffix\": \"_T12-5\"}" --force_compute True
```

