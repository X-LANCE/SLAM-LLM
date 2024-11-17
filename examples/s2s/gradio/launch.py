import gradio as gr
import soundfile as sf
from pydub import AudioSegment


def process_audio(audio_file):
    # 检查输入音频的格式
    if audio_file.endswith(".mp3"):
        audio = AudioSegment.from_mp3(audio_file)
        audio.export("temp_audio.wav", format="wav")
        audio_file = "temp_audio.wav"

    # 使用soundfile读取音频
    data, samplerate = sf.read(audio_file)
    
    # 模拟大模型推理过程（这里可以对音频数组进行处理）
    processed_data = data  # 假装做了一些复杂的处理
    
    # 将处理后的音频数组写回音频文件
    sf.write("processed_audio.wav", processed_data, samplerate)
    
    # 返回处理后的音频文件
    return "processed_audio.wav"

# 文本转音频函数：加载本地音频并模拟推理过程
def text_to_audio(text):
    # 加载本地音频文件 sample.wav
    data, samplerate = sf.read("sample.wav")
    
    # 模拟大模型推理过程
    processed_data = data  # 假装TTS
    
    # 将生成的音频写回文件
    sf.write("processed_text_audio.wav", processed_data, samplerate)
    
    # 返回处理后的音频文件
    return "processed_text_audio.wav"

# 创建Blocks页面
with gr.Blocks() as demo:
    # 第一部分：音频输入输出
    with gr.Row():
        gr.Markdown("### 输入音频，返回处理后音频")
        audio_input = gr.Audio(label="输入音频", type="filepath")
        audio_output = gr.Audio(label="输出音频")
        audio_button = gr.Button("处理音频")
        audio_button.click(process_audio, inputs=audio_input, outputs=audio_output)

    # 第二部分：文本输入，返回音频文件
    with gr.Row():
        gr.Markdown("### 输入文本，返回生成的本地音频文件")
        text_input = gr.Textbox(label="输入文本")
        file_output = gr.Audio(label="输出音频文件")
        text_button = gr.Button("生成音频")
        text_button.click(text_to_audio, inputs=text_input, outputs=file_output)

# 启动Gradio界面
demo.launch()
