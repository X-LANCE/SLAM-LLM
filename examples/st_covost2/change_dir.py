import os
import json

# 定义输入文件夹路径
folder_path = ""

# 定义关键词替换规则
old_keyword = ""  # 需要替换的关键词
new_keyword = "/code_dir"  # 替换成的关键词

# 遍历文件夹及其子文件夹
for root, _, files in os.walk(folder_path):
    for file_name in files:
        if file_name.endswith(".jsonl"):
            file_path = os.path.join(root, file_name)

            # 读取和处理 JSONL 文件
            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()

            updated_lines = []
            for line in lines:
                data = json.loads(line)
                if "audio" in data and old_keyword in data["audio"]:
                    data["audio"] = data["audio"].replace(old_keyword, new_keyword)
                updated_lines.append(json.dumps(data, ensure_ascii=False))

            # 写入修改后的内容到原文件
            with open(file_path, "w", encoding="utf-8") as file:
                file.write("\n".join(updated_lines))

            print(f"关键词替换完成，修改内容已写回文件: {file_path}")

print("所有文件处理完成。")