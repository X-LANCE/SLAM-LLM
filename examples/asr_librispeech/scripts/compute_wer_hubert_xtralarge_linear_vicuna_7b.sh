#!/bin/bash

output_dir=/home/v-yifyang/vicuna-7b-v1.5-hubert_xtralarge_ll60k_finetune_ls960
decode_log="decode_librispeech_test-clean_beam4"

cd ../..
python src/slam_llm/utils/whisper_tn.py ${output_dir}/${decode_log}_gt ${output_dir}/${decode_log}_gt.proc
python src/slam_llm/utils/whisper_tn.py ${output_dir}/${decode_log}_pred ${output_dir}/${decode_log}_pred.proc
python src/slam_llm/utils/compute_wer.py ${output_dir}/${decode_log}_gt.proc ${output_dir}/${decode_log}_pred.proc ${output_dir}/${decode_log}.proc.wer



