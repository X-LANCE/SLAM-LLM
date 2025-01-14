parent_dir=$1
gt_text="$parent_dir/gt_text"
pred_text="$parent_dir/pred_text"
pred_whisper="$parent_dir/pred_whisper_text"

python src/slam_llm/utils/whisper_tn.py ${gt_text} ${gt_text}.proc
python src/slam_llm/utils/whisper_tn.py ${pred_text} ${pred_text}.proc
python src/slam_llm/utils/compute_wer.py ${gt_text}.proc ${pred_text}.proc ${pred_text}.wer
# rm ${gt_text}.proc
# rm ${pred_text}.proc
tail -3 ${pred_text}.wer

# python src/slam_llm/utils/whisper_tn.py ${gt_text} ${gt_text}.proc
python src/slam_llm/utils/whisper_tn.py ${pred_whisper} ${pred_whisper}.proc
python src/slam_llm/utils/compute_wer.py ${gt_text}.proc ${pred_whisper}.proc ${pred_whisper}.wer
# rm ${gt_text}.proc
# rm ${pred_whisper}.proc
tail -3 ${pred_whisper}.wer