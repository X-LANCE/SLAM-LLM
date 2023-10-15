# Llama 2 Multimodal Fine-tuning / Inference Recipes and Examples

```python
# /path/to/transformers/src/transformers/modeling_utils.py:line 3108
if not isinstance(modules_to_not_convert, list):
    modules_to_not_convert = [modules_to_not_convert]

for name, module in model.named_children():
    if 'audio' in name or 'adapter' in name:
        keep_in_fp32_modules.append(name)
```