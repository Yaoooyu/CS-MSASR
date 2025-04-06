#!/bin/bash

# 模型路径
model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

# 输出目录
output_dir="/tmp/output/test/debug"

# 设备
device="cuda:0" # "cuda:0" for gpu0, "cuda:1" for gpu1, "cpu"

# 从test_wav.scp中读取每一行，提取第二个字段作为文件名
i=0
while IFS=' ' read -r _ input; do
    i=$((i+1))
    # 调用处理代码
    python -m funasr.bin.inference \
    ++model="${model}" \
    ++input="${input}" \
    ++output_dir="${output_dir}""$i" \
    ++device="${device}"
done < /home/liu/CS-MSASR/speech-recognize/list/test_wav.scp

