import sys
import os
import soundfile as sf
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/Matcha-TTS"))
from cosyvoice.cli.cosyvoice import CosyVoice,CosyVoice2
import torchaudio
import torch
import uuid

codec_decoder_path="/nfs/yangguanrou.ygr/ckpts/CosyVoice/CosyVoice-300M-SFT"
my_cosyvoice = CosyVoice("/nfs/yangguanrou.ygr/ckpts/CosyVoice/CosyVoice-300M-Instruct")

text="In the past perfect tense, the verb \"have\" is conjugated as \"had.\" For example: \"I had,\" \"You had,\" \"He had,\" \"She had,\" \"It had,\" \"We had,\" and \"They had.\""
text_new=my_cosyvoice.frontend.text_normalize(text)

# ['In the past perfect tense, the verb "have" is conjugated as "had." For example: "I had," "You had," "He had," "She had," "It had," "We had," and "They had."']
text1="If you\u2019re comparing items, consider checking reviews or ratings to understand if the higher price offers better value. "
text1_new=my_cosyvoice.frontend.text_normalize(text1)

text2="Three organizations that are actively working to combat global warming are the World Wildlife Fund (WWF), Greenpeace, and the Natural Resources Defense Council (NRDC)."
text2_new=my_cosyvoice.frontend.text_normalize(text2)

print(text_new)
print(text1_new)
print(text2_new)