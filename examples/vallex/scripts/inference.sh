python examples/vallex/inference_vallex.py \
--model_home "/home/wangtianrui/model_save/vallex_hf" \
--target_txt "读书使人进步,学习让我们的眼界更加开阔" \
--prompt_txt "笔记它只是一个工具就是最终的目的是吸收这些知识" \
--prompt_audio "examples/vallex/zh_prompt.wav"  \
--prompt_lang "zh"  \
--target_lang "zh" \
--save_path  "examples/vallex/test_out.wav" 
