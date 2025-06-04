from encodec.utils import convert_audio
from TTS.tts.layers.bark.hubert.hubert_manager import HubertManager
from TTS.tts.layers.bark.hubert.kmeans_hubert import CustomHubert
from TTS.tts.layers.bark.hubert.tokenizer import HubertTokenizer

import torchaudio


audio = "/Data3-Processing/nhandt23_bk/VoiceAnonymous/Attacker/test1.wav"

device = "cuda"

hubert_manager = HubertManager()
hubert_manager.make_sure_tokenizer_installed(model_path=self.model.config.LOCAL_MODEL_PATHS["hubert_tokenizer"])
hubert_model = CustomHubert(checkpoint_path=self.model.config.LOCAL_MODEL_PATHS["hubert"])  # .to(self.model.device)
tokenizer = HubertTokenizer.load_from_checkpoint(self.model.config.LOCAL_MODEL_PATHS["hubert_tokenizer"], map_location=device)


audio, sr = torchaudio.load(audio)
audio = convert_audio(audio, sr, self.model.config.sample_rate, self.model.encodec.channels)
audio = audio.to(device)  # there used to be an unsqueeze here but then they squeeze it back so it's useless

# 1. Extraction of semantic tokens
semantic_vectors = hubert_model.forward(audio, input_sample_hz=self.model.config.sample_rate)
semantic_tokens = tokenizer.get_token(semantic_vectors)
semantic_tokens = semantic_tokens.cpu().numpy() # they must be shifted to cpu