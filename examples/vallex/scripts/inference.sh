cd E:\\codes\\slam_new
PYTHONPATH=./ \
python examples/vallex/inference_vallex.py \
--model_home "E:\\data_models\\models\\vallex_hf_finetuned" \
--target_txt "读书使人进步,学习让我们的眼界更加开阔" \
--prompt_txt "笔记它只是一个工具就是最终的目的是吸收这些知识" \
--prompt_audio "examples/vallex/demo/zh_prompt.wav"  \
--prompt_lang "zh"  \
--target_lang "zh" \
--save_path  "examples/vallex/demo/zh2zh_test_out.wav" 


# cd E:\\codes\\slam_new
# PYTHONPATH=./ \
# python examples/vallex/inference_vallex.py \
# --model_home "E:\\data_models\\models\\vallex_hf_finetuned" \
# --target_txt "This is a versatile large model framework" \
# --prompt_txt "笔记它只是一个工具就是最终的目的是吸收这些知识" \
# --prompt_audio "examples/vallex/demo/zh_prompt.wav"  \
# --prompt_lang "zh"  \
# --target_lang "en" \
# --save_path  "examples/vallex/demo/zh2en_test_out.wav" 


# cd E:\\codes\\slam_new
# PYTHONPATH=./ \
# python examples/vallex/inference_vallex.py \
# --model_home "E:\\data_models\\models\\vallex_hf_finetuned" \
# --target_txt "This is a versatile large model framework" \
# --prompt_txt "BUT IN LESS THAN FIVE MINUTES THE STAIRCASE GROANED BENEATH AN EXTRAORDINARY WEIGHT" \
# --prompt_audio "examples/vallex/demo/en_prompt.flac"  \
# --prompt_lang "en"  \
# --target_lang "en" \
# --save_path  "examples/vallex/demo/en2en_test_out.wav" 


# cd E:\\codes\\slam_new
# PYTHONPATH=./ \
# python examples/vallex/inference_vallex.py \
# --model_home "E:\\data_models\\models\\vallex_hf_finetuned" \
# --target_txt "笔记它只是一个工具就是最终的目的是吸收这些知识" \
# --prompt_txt "BUT IN LESS THAN FIVE MINUTES THE STAIRCASE GROANED BENEATH AN EXTRAORDINARY WEIGHT" \
# --prompt_audio "examples/vallex/demo/en_prompt.flac"  \
# --prompt_lang "en"  \
# --target_lang "zh" \
# --save_path  "examples/vallex/demo/en2zh_test_out.wav" 