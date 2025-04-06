FunASR开源了大量在工业数据上预训练模型，您可以在 [模型许可协议](https://github.com/alibaba-damo-academy/FunASR/blob/main/MODEL_LICENSE)下自由使用、复制、修改和分享FunASR模型，下面列举代表性的模型，更多模型请参考 [模型仓库](https://github.com/alibaba-damo-academy/FunASR/tree/main/model_zoo)。

1. 该文件夹有3个文件：finetune.py, infer.py, infer-whisper.py
   finetune.py: 微调，使用模型openai/whisper-small
   infer.py: 使用finetune.py微调结果进行推理
   infer-whisper.py: 直接使用模型openai/whisper-small进行推理

2. FunASR源码中只有几个目录有finetune.sh，但whisper没有finetune.sh，仅有infer.sh，使用的模型是iic/speech_whisper-large_asr_multilingual。
   为了使用微调，我创建了finetune.py。
   FunASR源码中除了paraformer目录管用外，其他都不适合本数据集。例如fsmn_kws和sanm_kws要求音频文件有关键字，譬如将“小云小云”作为关键字来唤醒；如果用于我们的音频文件，则生成的文本为Rejected。如果音频文件中有“小云小云”，生成的文本应该是“小云小云”，而不是我们需要的文本。

3. finetune.py
   finetune.py使用模型openai/whisper-small，生成的模型结构如下:

ls /tmp/output:
added_tokens.json  merges.txt       preprocessor_config.json  tokenizer_config.json
checkpoint-545     normalizer.json  special_tokens_map.json   vocab.json

ls /tmp/output/checekpoint-545:
added_tokens.json       merges.txt         optimizer.pt              scheduler.pt             trainer_state.json
config.json             model.safetensors  preprocessor_config.json  special_tokens_map.json  training_args.bin
generation_config.json  normalizer.json    rng_state.pth             tokenizer_config.json    vocab.json

在finetune.py中设置save_total_limit=1，则在finetune.py运行过程中只生成最后1个checkpoint，该checkpoint即/tmp/output/checkpoint-545就是我们需要的模型，但它需要/tmp/output/preprocessor_config.json，所以可以将/tmp/output下的文件拷贝到/tmp/output/checkpoint-545。

infer.py最后会生成tokenTotal.txt，即把test_wav.scp测试集中的每个音频文件转录到tokenTotal.txt文件。

4. 本模型运行方法:
1）python finetune.py
   该程序生成一个/tmp/output目录，该目录下的checekpoint-545就是微调了的模型目录
2）python infer.py
   该程序会利用checekpoint-545中的模型，将其应用于测试集后生成文件tokenTotal.txt。对于每个音频文件而言，转录后的结果就是一句话（经过模型转录后与原文有一定区别）。
3) python infer-whisper.py
   该程序直接使用模型openai/whisper-small，应用于测试集后生成文件tokenTotal.txt。
