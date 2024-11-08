import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from tqdm import tqdm

def remove_outliers(data):
    """
    使用 IQR (Interquartile Range) 方法来去除离群点。
    """
    # 计算Q1和Q3
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    
    # 计算IQR
    IQR = Q3 - Q1
    
    # 计算离群点的上下限
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 过滤离群点
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    
    return filtered_data

def main(input_file):
    turn_counts = []

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            conversations = data.get("conversations", [])
            # 计算当前对话的轮数
            num_turns = len(conversations)/2
            if num_turns <= 10:
                turn_counts.append(num_turns)

    # 移除离群点
    # filtered_turn_counts = remove_outliers(turn_counts)

    # 统计每个轮数出现的频率
    count_dict = Counter(turn_counts)
    sorted_counts = sorted(count_dict.items())

    # 分离轮数和对应的对话数量
    turns, counts = zip(*sorted_counts)

    # 打印每个轮数的对话数量
    for turn, count in sorted_counts:
        print(f"turn: {turn}, count: {count}")

    # 可视化对话轮数的分布
    plt.figure(figsize=(10, 6))
    plt.bar(turns, counts, color='skyblue', edgecolor='black')
    plt.xlabel('turns')
    plt.ylabel('counts')
    plt.title('Distribution of turns')
    plt.xticks(turns)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('train_3.5M_CN_turns_distribution.png')


if __name__ == "__main__":
    input_file = '/mnt/bn/dev-mzy/data/corpus/belle_raw/train_3.5M_CN.json'
    main(input_file)
