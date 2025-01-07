#!/bin/bash
# Set up environment variables
export PYTHONPATH=/kaggle/working/SLAM-LLM:/kaggle/working/SLAM-LLM/src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# Chuyển đến thư mục làm việc Kaggle
cd /kaggle/working/SLAM-LLM

# Đường dẫn dữ liệu và mô hình
code_dir=examples/contextual_asr
speech_encoder_path=/kaggle/input/ckpts/wavlm_large_ft_libri960_char.pt
llm_path=/kaggle/input/vicuna-7b-v1.5
train_data_path=/kaggle/input/librispeech/librispeech_train_960h.jsonl
val_data_path=/kaggle/input/librispeech/librispeech_dev_other.jsonl
output_dir=/kaggle/working/output/vicuna-7b-v1.5-WavLM-Large-libri960-ft-char-hotwords-$(date +"%Y%m%d")

# Hydra arguments
hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=vicuna-7b-v1.5 \
++model_config.llm_path=$llm_path \
++model_config.encoder_path=$speech_encoder_path \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++train_config.output_dir=$output_dir \
"

# Huấn luyện với 2 GPU
torchrun --nnodes=1 --nproc-per-node=2 --master_port=29504 \
    $code_dir/finetune_contextual_asr.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    $hydra_args
