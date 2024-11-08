import re
import json
from zhon.hanzi import punctuation as zh_punctuation
from opencc import OpenCC
import string
from tqdm import tqdm

# 初始化OpenCC进行繁体到简体转换
cc = OpenCC('t2s')

# 定义合理的标点符号（中英文）
all_punctuation = string.punctuation + zh_punctuation + "×÷＝" + "°℃"

# 定义正则表达式模式
url_pattern = re.compile(r'https?://\S+|www\.\S+')
email_pattern = re.compile(r'\S+@\S+\.\S+')
code_pattern = re.compile(r'`{1,3}.*?`{1,3}', re.DOTALL)
emoji_pattern = re.compile(
    '['
    u'\U0001F600-\U0001F64F'  # 表情符号
    u'\U0001F300-\U0001F5FF'  # 符号和象形文字
    u'\U0001F680-\U0001F6FF'  # 交通和地图符号
    u'\U0001F1E0-\U0001F1FF'  # 国旗
    u'\U0001F700-\U0001F77F'  # 其他象形符号
    u'\U0001F780-\U0001F7FF'  # 更多符号
    u'\U0001F800-\U0001F8FF'  # 更多符号
    u'\U0001F900-\U0001F9FF'  # 更多表情符号
    u'\U0001FA00-\U0001FA6F'  # 角色面部表情、面具等
    u'\U0001FA70-\U0001FAFF'  # 高级符号，常见于现代设备
    u'\U00002702-\U000027B0'  # 符号（例如勾选、交叉）
    ']',
    re.UNICODE
)

table_pattern = re.compile(r'(\|.*\|(\n|\r\n)*)+')

file_patten = re.compile(r'\S+\.(txt|pdf|docx|doc|xlsx|xls|csv|pptx|ppt|png|jpg|jpeg|gif|mp3|mp4|wav|avi|mov|mkv|flv|wmv|zip|rar|tar|gz|7z|iso|dmg|pkg)')

phonetic_pattern = re.compile(r'/[a-zA-Zɪɛæʌʊɔəʌɜɑɔr]+/')  # 匹配 /i/、/ɪ/ 等音素符号

repeating_pattern = re.compile(r'[_\-*]{6,}')  # 匹配重复符号，如 ______

# 定义缩写词典
abbreviation_dict = {
    "Dr.": "Doctor",
    "e.g.": "for example",
    "i.e.": "that is",  # 常见的缩写，用于解释说明
    "Mr.": "Mister",
    "Mrs.": "Misses",
    "St.": "Saint",  # 用于地名或圣人的称谓
    "vs.": "versus",  # 用于对比，通常见于比赛或法庭用语
    "Prof.": "Professor",
    "Ave.": "Avenue",
    "Dept.": "Department",
    "etc.": "and so on",  # 表示等等
    "Inc.": "Incorporated",  # 公司名中常见
    "Ltd.": "Limited",  # 有限公司
    "Jr.": "Junior",
    "Sr.": "Senior",
    "vs.": "versus",  # 对抗
    "approx.": "approximately",  # 大约
    "min.": "minute",  # 分钟
    "sec.": "second",  # 秒
    "Fri.": "Friday",
    "Sat.": "Saturday",
    "Sun.": "Sunday",
    "Mon.": "Monday",
    "Tue.": "Tuesday",
    "Wed.": "Wednesday",
    "Thu.": "Thursday",
    "no.": "number",  # 常用于序号
    "No.": "number",
    "Jan.": "January",
    "Feb.": "February",
    "Mar.": "March",
    "Apr.": "April",
    "Jun.": "June",
    "Jul.": "July",
    "Aug.": "August",
    "Sept.": "September",
    "Oct.": "October",
    "Nov.": "November",
    "Dec.": "December",
    "est.": "established",  # 成立于（例如用于公司成立时间）
    "max.": "maximum",  # 最大值
    "min.": "minimum",  # 最小值
}

def normalize_abbreviations(text):
    for abbr, full in abbreviation_dict.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text)
    return text

# 定义要过滤的敏感词列表（不包含具体词汇）
sensitive_words = set()  # 可以在这里添加敏感词汇

def filter_inappropriate_content(text):
    # 可以使用外部库或自定义方法过滤不当内容
    # 这里只是示例，不包含具体实现
    for word in sensitive_words:
        text = text.replace(word, '')
    return text

def process_text(text):
    # 将繁体中文转换为简体中文
    text = cc.convert(text)
    
    # 规范缩写
    text = normalize_abbreviations(text)
    
    # 过滤不当内容
    text = filter_inappropriate_content(text)
    
    return text

def is_valid_char(text):
    return all(
        (u'\u4e00' <= char <= u'\u9fff')  # 保留汉字
        or char.isalpha()  # 保留字母
        or char in all_punctuation  # 保留标点符号
        or char.isspace()  # 允许空格
        or char == '\n'  # 允许换行符
        or char.isdigit()  # 允许数字
        for char in text
    )

def is_valid_sentence(sentence):
    # 检查URL和链接
    if url_pattern.search(sentence):
        return False, "URL"

    # 检查电子邮件地址
    if email_pattern.search(sentence):
        return False, "Email"
    
    # 检查代码片段
    if code_pattern.search(sentence):
        return False, "Code"
    
    # 检查表情符号
    if emoji_pattern.search(sentence):
        return False, "Emoji"
    
    # 检查表格
    if table_pattern.search(sentence):
        return False, "Table"
    
    # 检查文件
    if file_patten.search(sentence):
        return False, "File"
    
    # 检查音素符号
    if phonetic_pattern.search(sentence):
        return False, "Phonetic"
    
    # 检查重复符号
    if repeating_pattern.search(sentence):
        return False, "Repeating"
    
    return True, "Valid"
    
def suitable_for_tts(data, max_turn_num=10, max_text_length=200):
    # 检查对话轮数
    if len(data["conversations"]) > max_turn_num:
        return False
    
    # 检查文本长度
    for conversation in data["conversations"]:
        if len(conversation["value"]) > max_text_length:
            return False

    return True

def process_conversations(input_file, output_file, invalid_file, tts_file):

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile, open(invalid_file, 'w', encoding='utf-8') as invalidfile, open(tts_file, 'w', encoding='utf-8') as ttsfile:
        
        valid_num = 0
        invalid_num = 0
        tts_num = 0

        # 流式处理，逐行读取和处理
        for line in tqdm(infile):
            data = json.loads(line)

            # 处理对话数据
            processed_data = {
                "conversations": [],
                "id": data.get("id", "")
            }
            valid = True  # 标记是否有有效的对话

            for conversation in data.get("conversations", []):
                text = conversation.get("value", "").strip()

                if not is_valid_char(text):
                    # print(f"Invalid Character error in: {text}")
                    valid = False
                    invalid_num += 1
                    data["invalid_reason"] = "Character"
                    json.dump(data, invalidfile, ensure_ascii=False)
                    invalidfile.write('\n')
                    break

                valid_sentence, reason = is_valid_sentence(text)
                if not valid_sentence:
                    # print(f"{reason} error in: {text}")
                    valid = False
                    invalid_num += 1
                    data["invalid_reason"] = reason
                    json.dump(data, invalidfile, ensure_ascii=False)
                    invalidfile.write('\n')
                    break


                # 处理文本
                processed_text = process_text(text)
                if not is_valid_char(processed_text) or not is_valid_sentence(processed_text):
                    valid = False
                    break
                
                # 构建新的对话结构
                processed_conversation = {
                    "from": conversation.get("from", ""),
                    "value": processed_text
                }
                processed_data["conversations"].append(processed_conversation)
            
            if valid:
                valid_num += 1
                json.dump(processed_data, outfile, ensure_ascii=False)
                outfile.write('\n')  # 添加换行符，以便下一行开始

                if suitable_for_tts(processed_data, 10, 200):
                    tts_num += 1
                    json.dump(processed_data, ttsfile, ensure_ascii=False)
                    ttsfile.write('\n')  # 添加换行符，以便下一行开始


        print(f"Total number of valid conversations: {valid_num}")
        print(f"Total number of invalid conversations: {invalid_num}")
        print(f"Total number of conversations suitable for TTS: {tts_num}")
        

if __name__ == "__main__":
    # 输入和输出文件路径
    input_file = '/mnt/bn/dev-mzy/data/corpus/belle_raw/train_3.5M_CN.json'   # 原始对话数据文件
    output_file = '/mnt/bn/dev-mzy/data/corpus/belle_raw/train_3.5M_CN_cleaned.json' # 处理后的数据文件
    invalid_file = '/mnt/bn/dev-mzy/data/corpus/belle_raw/train_3.5M_CN_invalid.json'  # 无效对话数据文件
    tts_file = '/mnt/bn/dev-mzy/data/corpus/belle_raw/train_3.5M_CN_ready4cosy.json' # 准备好的tts数据文件

    process_conversations(input_file, output_file, invalid_file, tts_file)
    # test = {"conversations": [{"from": "human", "value": "生成一个调查问卷，调查大学生对校园环境改善的看法和建议。"}, {"from": "assistant", "value": "调查问卷：大学生对校园环境改善的看法和建议\n你好，我们正在进行一项调查，旨在了解大学生们对于校园环境改善的看法和建议。希望您能抽出几分钟时间填写这份问卷，谢谢！\n1. 你是哪所大学的学生？\nA. xx大学\nB. xx大学\nC. 其他（请注明：________）\n2. 你认为你所在的学校的校园环境如何？\nA. 很好 B. 较好 C. 一般 D. 较差 E. 很差\n3. 对于学校的校园环境，你最满意的是什么方面？\nA. 绿化和景观\nB. 卫生和清洁\nC. 建筑和设施\nD. 安全和保障\nE. 其他（请注明：________）\n4. 对于学校的校园环境，你最不满意的是什么方面？\nA. 绿化和景观\nB. 卫生和清洁\nC. 建筑和设施\nD. 安全和保障\nE. 其他（请注明：________）\n5. 你认为有哪些方面可以改善你所在学校的校园环境？（请勾选）\nA. 增加绿化和植树\nB. 加大垃圾分类投放力度\nC. 加强设施和建筑维护\nD. 提高校园安全保障\nE. 加大环保宣传和教育力度\nF. 其他（请注明：________）\n6. 你觉得学校和学生可以共同做些什么来改善校园环境？\nA. 学生可以更爱护校园环境\nB. 学校可以加强环境管理和维护力度\nC. 设置环保奖励机制鼓励环保行为\nD. 加强师生环保教育\nE. 其他（请注明：________）\n7. 你认为学生在校内生态保护方面需要加强哪些方面的意识和行为？\nA. 节约用水和用电\nB. 垃圾分类和投放\nC. 自觉保护校园环境\nD. 拒绝乱扔垃圾和破坏设施\nE. 其他（请注明：________）\n8. 如果你可以对你所在学校的校园环境进行改善，你会怎么做？\nA. 增加植物和花草，美化校园\nB. 加强清洁和卫生管理\nC. 提高设施和建筑维护水平\nD. 加强安保力度保障学生个人安全\nE. 向学生提供更多与环保有关的活动和项目\nF. 其他（请注明：________）\n9. 最后，你认为校园环境的改善对于学生学习和生活的影响是什么？\nA. 可以提高学生的学习积极性和创造力\nB. 可以改善学生的生活和健康状况\nC. 可以提高学生的环保意识和社会责任感\nD. 可以促进学生个人成长和发展\nE. 其他（请注明：________）\n感谢您的参与！"}], "id": "18054991"}
    # for conversations in test["conversations"]:
    #     print(conversations["value"])
    #     print(is_valid_char(conversations["value"]))
    #     print(is_valid_sentence(conversations["value"]))
