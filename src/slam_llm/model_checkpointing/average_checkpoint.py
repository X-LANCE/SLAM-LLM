import os
import torch
import argparse
from safetensors.numpy import load_file, save_file

def average_models(args):
    # 创建新目录
    os.makedirs(args.output, exist_ok=True)

    # 加载model.pt
    checkpoint_list = args.input.split(",")
    model_list = []
    for dir in checkpoint_list:
        model_list.append(torch.load(os.path.join(dir, 'model.pt')))
    num = len(model_list)

    # 加载adapter
    if args.peft:
        adapter_list = []
        for dir in checkpoint_list:
            adapter_list.append(load_file(os.path.join(dir, "adapter_model.safetensors")))

    # 对模型参数取平均
    averaged_model = {}
    for key in model_list[0].keys():
        averaged_model[key] = model_list[0][key]
        for i in range(1,num):
            averaged_model[key] += model_list[i][key]
        averaged_model[key] /= num

    # 对适配器模型参数取平均
    if args.peft:
        averaged_adapter = {}
        for key in adapter_list[0].keys():
            averaged_adapter[key] = adapter_list[0][key]
            for i in range(1,num):
                averaged_adapter[key] += adapter_list[i][key]
            averaged_adapter[key] /= num

    # 保存平均后的模型和适配器模型
    torch.save(averaged_model, os.path.join(args.output, 'model.pt'))
    if args.peft:
        save_file(averaged_adapter, os.path.join(args.output, "adapter_model.safetensors"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="checkpoint dir")
    parser.add_argument("--output", type=str, help="Output dir")
    parser.add_argument("--peft", action="store_true", help="Whether to use peft")
    args = parser.parse_args()
    average_models(args)

if __name__ == "__main__":
    main()
