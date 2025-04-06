# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

workspace=`pwd`

# which gpu to train or finetune
export CUDA_VISIBLE_DEVICES="0"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

#model_name_or_model_dir="/home/liu/.cache/modelscope/hub/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
model_name_or_model_dir="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

# data dir, which contains: train.json, val.json
data_dir="/home/liu/CS-MSASR/speech-recognize/list"

train_data="${data_dir}/train.jsonl"
val_data="${data_dir}/val.jsonl"

rm -f ${train_data}
rm -f ${val_data}

# generate train.jsonl and val.jsonl from wav.scp and text.txt
scp2jsonl \
++scp_file_list='["/home/liu/CS-MSASR/speech-recognize/list/train_wav.scp", "/home/liu/CS-MSASR/speech-recognize/list/train_text.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="${train_data}"

scp2jsonl \
++scp_file_list='["/home/liu/CS-MSASR/speech-recognize/list/val_wav.scp", "/home/liu/CS-MSASR/speech-recognize/list/val_text.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="${val_data}"

# exp output dir
output_dir="/tmp/output"
log_file="${output_dir}/log.txt"
rm -rf ${output_dir}

deepspeed_config=${workspace}/../../ds_stage1.json

mkdir -p ${output_dir}
echo "log_file: ${log_file}"

DISTRIBUTED_ARGS="
    --nnodes ${WORLD_SIZE:-1} \
    --nproc_per_node $gpu_num \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-26669}
"

echo $DISTRIBUTED_ARGS

#You should create a python virtual environment and pip install pytorch, funasr, ... in /home/liu/CS-MSASR/speech-recognize/
#python3 -m venv pytorch-env
torchrun $DISTRIBUTED_ARGS \
/home/liu/CS-MSASR/speech-recognize/pytorch-env/lib/python3.12/site-packages/funasr/bin/train_ds.py \
++model="${model_name_or_model_dir}" \
++train_data_set_list="${train_data}" \
++valid_data_set_list="${val_data}" \
++dataset="AudioDataset" \
++dataset_conf.index_ds="IndexDSJsonl" \
++dataset_conf.data_split_num=1 \
++dataset_conf.batch_sampler="BatchSampler" \
++dataset_conf.batch_size=6000  \
++dataset_conf.sort_size=1024 \
++dataset_conf.batch_type="token" \
++dataset_conf.num_workers=4 \
++train_conf.max_epoch=50 \
++train_conf.log_interval=1 \
++train_conf.resume=true \
++train_conf.validate_interval=2000 \
++train_conf.save_checkpoint_interval=2000 \
++train_conf.keep_nbest_models=20 \
++train_conf.avg_nbest_model=10 \
++train_conf.use_deepspeed=false \
++train_conf.deepspeed_config=${deepspeed_config} \
++optim_conf.lr=0.0002 \
++output_dir="${output_dir}" &> ${log_file}
