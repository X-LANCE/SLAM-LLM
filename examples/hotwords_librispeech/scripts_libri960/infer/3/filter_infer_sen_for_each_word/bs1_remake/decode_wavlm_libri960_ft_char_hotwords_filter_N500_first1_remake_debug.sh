#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

run_dir=/root/SLAM-LLM
cd $run_dir
code_dir=examples/hotwords_librispeech


speech_encoder_path=/nfs/yangguanrou.ygr/ckpts/wavlm_large_ft_libri960_char/wavlm_large_ft_libri960_char.pt
llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5

output_dir=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-libri960-ft-char-hotwords-20240521
ckpt_path=$output_dir/asr_epoch_3_step_25780

first=1
for N in 500; do
        for ref_split in test_clean test_other; do
                split=librispeech_${ref_split}
                val_data_path=/nfs/maziyang.mzy/data/librispeech/${split}.jsonl
                decode_log=$ckpt_path/decode_${split}_beam4_filter_N${N}_first${first}_remakedebug
                python $code_dir/inference_asr_batch.py \
                        --config-path "conf" \
                        --config-name "prompt.yaml" \
                        hydra.run.dir=$ckpt_path \
                        ++model_config.llm_name="vicuna-7b-v1.5" \
                        ++model_config.llm_path=$llm_path \
                        ++model_config.llm_dim=4096 \
                        ++model_config.encoder_name=wavlm \
                        ++model_config.normalize=true \
                        ++dataset_config.normalize=true \
                        ++model_config.encoder_projector_ds_rate=5 \
                        ++model_config.encoder_path=$speech_encoder_path \
                        ++model_config.encoder_dim=1024 \
                        ++model_config.encoder_projector=cov1d-linear \
                        ++dataset_config.dataset=speech_dataset \
                        ++dataset_config.val_data_path=$val_data_path \
                        ++dataset_config.input_type=raw \
                        ++dataset_config.inference_mode=true \
                        ++dataset_config.infer_type=filter \
                        ++dataset_config.dataset=hotwordsinfer_dataset \
                        ++dataset_config.file=src/slam_llm/datasets/hotwordsinfer_dataset.py:get_speech_dataset \
                        ++dataset_config.infer_file=/nfs/yangguanrou.ygr/data/fbai-speech/is21_deep_bias/my_ref/${ref_split}.biasing_${N}.tsv \
                        ++dataset_config.ctc_file=/nfs/yangguanrou.ygr/data/librispeech_my_infer/wavlm_ft_libri960_${ref_split}_char.txt \
                        ++dataset_config.probability_threshold=0.9 \
                        ++dataset_config.word_num=15 \
                        ++dataset_config.filter_infer_sentence=true \
                        ++dataset_config.filter_infer_sentence_few=true \
                        ++train_config.model_name=asr \
                        ++train_config.freeze_encoder=true \
                        ++train_config.freeze_llm=true \
                        ++train_config.batching_strategy=custom \
                        ++train_config.num_epochs=1 \
                        ++train_config.val_batch_size=1 \
                        ++train_config.num_workers_dataloader=0 \
                        ++train_config.output_dir=$output_dir \
                        ++decode_log=$decode_log \
                        ++ckpt_path=$ckpt_path/model.pt && \

                python src/slam_llm/utils/whisper_tn.py ${decode_log}_gt ${decode_log}_gt.proc && \
                python src/slam_llm/utils/whisper_tn.py ${decode_log}_pred ${decode_log}_pred.proc && \
                python src/slam_llm/utils/compute_wer.py ${decode_log}_gt.proc ${decode_log}_pred.proc ${decode_log}.proc.wer && \
                python /nfs/yangguanrou.ygr/data/fbai-speech/is21_deep_bias/my_score.py \
                        --refs /nfs/yangguanrou.ygr/data/fbai-speech/is21_deep_bias/ref_score/${ref_split}.biasing_${N}.tsv \
                        --hyps ${decode_log}_pred.proc \
                        --output_file ${decode_log}.proc.wer
        done
done


# bash /root/SLAM-LLM/examples/hotwords_librispeech/scripts_libri960/infer/3/filter_infer_sen_for_each_word/bs1_remake/decode_wavlm_libri960_ft_char_hotwords_filter_N500_first1_remake_debug.sh > /root/SLAM-LLM/examples/hotwords_librispeech/scripts_libri960/infer/3/filter_infer_sen_for_each_word/bs1_remake/decode_wavlm_libri960_ft_char_hotwords_filter_N500_first1_remake_debug.log