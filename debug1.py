# from whisper.normalizers import EnglishTextNormalizer

# from whisper.normalizers.english import EnglishNumberNormalizer

from whisper.normalizers.english import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()
# normalizer1 = EnglishNumberNormalizer()


# with open("/root/SLAM-LLM/slides_script/3.1/whisper_decode_results/my_tn/decode_dev_whisper_1_pred",'r') as f:
#     with open("/root/SLAM-LLM/slides_script/3.1/whisper_decode_results/my_tn/decode_dev_whisper_1_pred_tn",'w') as f1:
#         for line in f:
#             line=line.split('\t')
#             if line[0] == "YTB+--tMoLpQI-w+00012" or line[0] == "YTB+yXWQtSspq7Q+00138" or line[0] == "YTB+DVLSyKTp6dk+00141":
#                 continue

#             line[1]=normalizer(line[1])
#             # line[1]=normalizer1(line[1])
#             f1.write(line[0]+'\t'+line[1]+'\n')


# with open("/root/SLAM-LLM/slides_script/3.1/whisper_decode_results/my_tn/decode_dev_whisper_1_gt",'r') as f:
#     with open("/root/SLAM-LLM/slides_script/3.1/whisper_decode_results/my_tn/decode_dev_whisper_1_gt_tn",'w') as f1:
#         for line in f:
#             line=line.split('\t')
#             if line[0] == "YTB+--tMoLpQI-w+00012" or line[0] == "YTB+yXWQtSspq7Q+00138" or line[0] == "YTB+DVLSyKTp6dk+00141":
#                 continue

#             line[1]=normalizer(line[1])
#             # line[1]=normalizer1(line[1])
#             f1.write(line[0]+'\t'+line[1]+'\n')



with open("/nfs/yangguanrou.ygr/slides-finetune-whisperv3/asr/9760/mytn/decode_log_dev_clean_beam4_repetition_penalty1_pred",'r') as f:
    with open("/nfs/yangguanrou.ygr/slides-finetune-whisperv3/asr/9760/mytn/decode_log_dev_clean_beam4_repetition_penalty1_pred_tn",'w') as f1:
        for line in f:
            line=line.split('\t')
            if line[0] == "YTB+--tMoLpQI-w+00012" or line[0] == "YTB+yXWQtSspq7Q+00138" or line[0] == "YTB+DVLSyKTp6dk+00141":
                continue

            line[1]=normalizer(line[1])
            # line[1]=normalizer1(line[1])
            f1.write(line[0]+'\t'+line[1]+'\n')


with open("/nfs/yangguanrou.ygr/slides-finetune-whisperv3/asr/9760/mytn/decode_log_dev_clean_beam4_repetition_penalty1_gt",'r') as f:
    with open("/nfs/yangguanrou.ygr/slides-finetune-whisperv3/asr/9760/mytn/decode_log_dev_clean_beam4_repetition_penalty1_gt_tn",'w') as f1:
        for line in f:
            line=line.split('\t')
            if line[0] == "YTB+--tMoLpQI-w+00012" or line[0] == "YTB+yXWQtSspq7Q+00138" or line[0] == "YTB+DVLSyKTp6dk+00141":
                continue

            line[1]=normalizer(line[1])
            # line[1]=normalizer1(line[1])
            f1.write(line[0]+'\t'+line[1]+'\n')



# EnglishTextNormalizer 包含这俩
# self.standardize_numbers = EnglishNumberNormalizer()
# self.standardize_spellings = EnglishSpellingNormalizer()

# 应该有1798句  去掉三句纯UM

# EnglishTextNormalizer() 这个就够用