# Llama 2 Multimodal Fine-tuning / Inference Recipes and Examples

## install dependences
```bash
# huggingface transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .

# peft 
git clone https://github.com/huggingface/peft.git
cd peft
pip install -e .

git clone https://github.com/zszheng147/llama-recipes.git
cd llama-recipes
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 -e .
```

## Remain audio modules fixed
```python
# modify /path/to/transformers/src/transformers/modeling_utils.py:line 3108
if not isinstance(modules_to_not_convert, list):
    modules_to_not_convert = [modules_to_not_convert]

for name, module in model.named_children():
    if 'audio' in name:
        keep_in_fp32_modules.append(name)
```

## Install backbone models
```text
audioMAE (encoder): https://drive.google.com/file/d/18EsFOyZYvBYHkJ7_n7JFFWbj6crz01gq/view?usp=share_link
llama-2: https://ai.meta.com/resources/models-and-libraries/llama-downloads/
```

## Model conversion to Hugging Face
```bash
cd transformers
pip install protobuf
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
   --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

## Train
in scripts/single_finetune.sh


replace *audio_encoder_path* and *model_name* with your path
```bash
bash scripts/single_finetune.sh
```