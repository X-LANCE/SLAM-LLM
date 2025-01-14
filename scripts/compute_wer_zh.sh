parent_dir=$1
gt_text="$parent_dir/gt_text"
pred_text="$parent_dir/pred_text"
pred_whisper="$parent_dir/pred_whisper_text"

python src/slam_llm/utils/whisper_tn.py ${gt_text} ${gt_text}.proc
python src/slam_llm/utils/whisper_tn.py ${pred_text} ${pred_text}.proc
python src/slam_llm/utils/compute_wer_zh.py ${gt_text}.proc ${pred_text}.proc ${pred_text}.wer

python src/slam_llm/utils/compute_wer_zh.py ${gt_text} ${pred_text} ${pred_text}.onlywer
# rm ${gt_text}.proc
# rm ${pred_text}.proc
tail -3 ${pred_text}.wer
tail -3 ${pred_text}.onlywer

# python src/slam_llm/utils/whisper_tn.py ${gt_text} ${gt_text}.proc
python src/slam_llm/utils/whisper_tn.py ${pred_whisper} ${pred_whisper}.proc
python src/slam_llm/utils/compute_wer_zh.py ${gt_text}.proc ${pred_whisper}.proc ${pred_whisper}.wer

python src/slam_llm/utils/compute_wer_zh.py ${gt_text} ${pred_whisper} ${pred_whisper}.onlywer
# rm ${gt_text}.proc
# rm ${pred_whisper}.proc
tail -3 ${pred_whisper}.wer

tail -3 ${pred_whisper}.onlywer
#  -m debugpy --listen 5678 --wait-for-client 