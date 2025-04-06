# Copyright © by Yaoyu Liu
import argparse
import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa

class WhisperDataProcessor:
    def __init__(self, processor):
        self.processor = processor

    def process_audio(self, audio_path):
        # 确保文件存在
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件 {audio_path} 不存在")
        
        # 读取音频并转换为16kHz
        waveform, _ = librosa.load(audio_path, sr=16000)
        
        # 提取特征
        inputs = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt"
        )
        input_features = inputs.input_features.squeeze(0)  # [80, 时间步长]
        
        # 生成注意力掩码
        attention_mask = (input_features.abs().sum(dim=0) != 0).long()  # [时间步长]
        
        # 填充或截断到 3000 时间步长（与训练时保持一致）
        max_mel_len = 3000
        if input_features.shape[1] > max_mel_len:
            input_features = input_features[:, :max_mel_len]  # 截断
            attention_mask = attention_mask[:max_mel_len]
        else:
            padding = torch.zeros((80, max_mel_len - input_features.shape[1]))
            input_features = torch.cat([input_features, padding], dim=1)  # 填充
            padding_mask = torch.zeros(max_mel_len - input_features.shape[1]).long()
            attention_mask = torch.cat([attention_mask, padding_mask], dim=0)
        
        # 添加批次维度
        input_features = input_features.unsqueeze(0)  # [1, 80, 3000]
        attention_mask = attention_mask.unsqueeze(0)  # [1, 3000]
        
        return input_features, attention_mask

def transcribe_audio(model, processor, audio_path):
    data_processor = WhisperDataProcessor(processor)
    input_features, attention_mask = data_processor.process_audio(audio_path)
    
    # 将输入转换为 float16
    input_features = input_features.half()
    
    # 转录音频
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            task="transcribe",  # 明确设置任务为转录
            language="zh",      # 明确设置语言为中文
            forced_decoder_ids=None,  # 明确移除 forced_decoder_ids
        )
    
    # 解码转录文本
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="openai/whisper-small", help="训练好的模型检查点路径")
    args = parser.parse_args()

    # 加载模型和处理器
    processor = WhisperProcessor.from_pretrained(args.model_checkpoint)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_checkpoint, torch_dtype=torch.float16)

    # 显式移除 forced_decoder_ids 的默认值
    model.generation_config.forced_decoder_ids = None

    with open('/home/liu/CS-MSASR/speech-recognize/list/test_wav.scp', 'r') as file:
        with open('tokenTotal.txt', 'w') as output_file:
            for line in file:
                # 去除行末的换行符并分割成两个字段
                parts = line.strip().split()
                if len(parts) == 2:
                    # 分两行打印两个字段
                    first = parts[0]
                    audio_file = parts[1]

                    # 转录音频文件
                    try:
                        transcription = transcribe_audio(model, processor, audio_file)
                    
                        print("音频文字: ", transcription)
                        output_line = f"{first} {transcription}\n"
                
                        # 写入文件
                        output_file.write(output_line)
                        # 强制将缓冲区内容写入文件
                        output_file.flush()  
                    except Exception as e:
                        print(f"Error during transcription: {e}")

                else:
                    print(f"跳过无效行: {line.strip()}")

