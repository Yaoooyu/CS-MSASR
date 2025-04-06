# Copyright © by Yaoyu Liu
import argparse
import librosa
import os
import torch
from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Trainer,
    TrainingArguments,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class WhisperDataCollator:
    def __init__(self, processor: Any):
        self.processor = processor
 
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [
            torch.tensor(feature["input_features"]) if isinstance(feature["input_features"], list) else feature["input_features"]
            for feature in features
        ]
        input_features = torch.stack(input_features)
        attention_mask = (input_features.abs().sum(dim=2) != 0).long()
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(feature["labels"]) for feature in features],
            batch_first=True,
            padding_value=-100
        )
        return {
            "input_features": input_features,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def prepare_dataset(batch, processor):
    audio_path = batch["source"]
    text = batch["target"]
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件 {audio_path} 不存在")
    waveform, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt"
    )
    input_features = inputs.input_features.squeeze(0)
    max_mel_len = 3000
    if input_features.shape[1] > max_mel_len:
        input_features = input_features[:, :max_mel_len]
    else:
        padding = torch.zeros((80, max_mel_len - input_features.shape[1]))
        input_features = torch.cat([input_features, padding], dim=1)
    labels = processor.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.squeeze(0)
    return {
        "input_features": input_features,
        "labels": labels
    }

if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="openai/whisper-small")
    parser.add_argument("--train_dataset", type=str, default="/home/liu/CS-MSASR/speech-recognize/list/train.jsonl")
    parser.add_argument("--eval_dataset", type=str, default="/home/liu/CS-MSASR/speech-recognize/list/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="/tmp/output")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    processor = WhisperProcessor.from_pretrained(args.model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)

    # 显式设置 use_cache = False 以避免与梯度检查点的不兼容问题
    model.config.use_cache = False

    train_dataset = load_dataset("json", data_files={"train": args.train_dataset}, split="train")
    eval_dataset = load_dataset("json", data_files={"validation": args.eval_dataset}, split="validation")
    
    train_dataset = train_dataset.map(
        prepare_dataset,
        fn_kwargs={"processor": processor},
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        prepare_dataset,
        fn_kwargs={"processor": processor},
        remove_columns=eval_dataset.column_names
    )

    data_collator = WhisperDataCollator(processor=processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        fp16=args.fp16,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    processor.save_pretrained(args.output_dir)