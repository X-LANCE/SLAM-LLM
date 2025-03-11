#!/bin/bash
# 让脚本并行地在 8 张卡上跑推理，每张卡处理 1/8 的数据。

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1

code_dir=examples/s2s

speech_encoder_path="small"   # whisper small
llm_path="/nfs/yangguanrou.ygr/ckpts/Qwen/Qwen2.5-0.5B"
codec_decoder_path="/nfs/yangguanrou.ygr/ckpts/CosyVoice/CosyVoice-300M-SFT"

tts_adapter=false
task_type=tts

ckpt_path=/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/scripts/ygr/exp/tts/ft_secap_qwen25_lr5e-6_50k/s2s_epoch_8_step_2677
split=test

# jsonl dataset
manifest_format=jsonl


# vocabulary settings
code_layer=3
total_audio_vocabsize=4160
total_vocabsize=156160

# code settings
code_type=CosyVoice
codec_decoder_type=CosyVoice
num_latency_tokens=5
do_layershift=false

# model settings
group_decode=true
group_decode_adapter_type=linear

# decode config
text_repetition_penalty=1.2
audio_repetition_penalty=1.2
max_new_tokens=3000
do_sample=false
top_p=1.0
top_k=0
temperature=1.0
decode_text_only=false

output_text_only=false
inference_online=false
speech_sample_rate=22050

decode_log=$ckpt_path/tts_decode_${split}_rp${repetition_penalty}_seed${dataset_sample_seed}_greedy_train_single_0_8000

# 如果 do_sample=true，可自行修改 decode_log 的名字

# 这里用一个 for 循环，遍历 shard_index = 0..7
# 每个进程使用不同的 CUDA_VISIBLE_DEVICES，从而在 8 张卡上并行推理。

for i in {0..3}
do
    val_data_path=/nfs/yangguanrou.ygr/data/secap_my/dpo/first_trial/train_single_0_8000.${i}
  # 后台并行执行的话，可以在末尾加个 &，然后再统一 wait
  # 如果想串行执行，就去掉 & 和 wait
    CUDA_VISIBLE_DEVICES=$i python $code_dir/inference_s2s.py \
        --config-path "conf" \
        --config-name "prompt_tts.yaml" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name=qwen2-0.5b \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=896 \
        ++model_config.encoder_name=whisper \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=768 \
        ++model_config.encoder_projector=linear \
        ++model_config.codec_decoder_path=$codec_decoder_path \
        ++model_config.codec_decode=true \
        ++model_config.vocab_config.code_layer=$code_layer \
        ++model_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
        ++model_config.vocab_config.total_vocabsize=$total_vocabsize \
        ++model_config.code_type=$code_type \
        ++model_config.codec_decoder_type=$codec_decoder_type \
        ++model_config.group_decode=$group_decode \
        ++model_config.group_decode_adapter_type=$group_decode_adapter_type \
        ++model_config.save_audio_token=true \
        ++dataset_config.dataset=speech_dataset_s2s \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.train_data_path=$val_data_path \
        ++dataset_config.input_type=mel \
        ++dataset_config.mel_size=80 \
        ++dataset_config.inference_mode=true \
        ++dataset_config.manifest_format=$manifest_format \
        ++dataset_config.task_type=$task_type \
        ++dataset_config.vocab_config.code_layer=$code_layer \
        ++dataset_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
        ++dataset_config.vocab_config.total_vocabsize=$total_vocabsize \
        ++dataset_config.code_type=$code_type \
        ++dataset_config.num_latency_tokens=$num_latency_tokens \
        ++dataset_config.do_layershift=$do_layershift \
        ++dataset_config.use_emo=true \
        ++train_config.model_name=s2s \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.freeze_group_decode_adapter=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.task_type=$task_type \
        ++decode_config.text_repetition_penalty=$text_repetition_penalty \
        ++decode_config.audio_repetition_penalty=$audio_repetition_penalty \
        ++decode_config.max_new_tokens=$max_new_tokens \
        ++decode_config.task_type=$task_type \
        ++decode_config.do_sample=$do_sample \
        ++decode_config.top_p=$top_p \
        ++decode_config.top_k=$top_k \
        ++decode_config.temperature=$temperature \
        ++decode_config.decode_text_only=$decode_text_only \
        ++decode_config.num_latency_tokens=$num_latency_tokens \
        ++decode_config.do_layershift=$do_layershift \
        ++decode_log=${decode_log}_shard$i \
        ++ckpt_path=$ckpt_path/model.pt \
        ++output_text_only=$output_text_only \
        ++inference_online=$inference_online \
        ++speech_sample_rate=$speech_sample_rate &

done
