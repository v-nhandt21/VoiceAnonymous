from speechbrain.inference.interfaces import foreign_class
classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
out_prob, score, index, text_lab = classifier.classify_file("/data1/nhandt23/VoicePrivacy/Attacker/test1.wav")
print(text_lab)


# Installing collected packages: speechbrain
#   Attempting uninstall: speechbrain
#     Found existing installation: speechbrain 0.5.16
#     Uninstalling speechbrain-0.5.16:
#       Successfully uninstalled speechbrain-0.5.16
# Successfully installed speechbrain-1.0.2