#!/bin/bash
# Set up environment variables
export PYTHONPATH=/kaggle/working/SLAM-LLM:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1  # Sử dụng 2 GPU
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# Chuyển đến thư mục làm việc Kaggle
cd /kaggle/working/SLAM-LLM

# Đường dẫn dữ liệu và mô hình trên Kaggle
code_dir=examples/contextual_asr

speech_encoder_path=/kaggle/input/ckpts/wavlm_large_ft_libri960_char.pt
llm_path=/kaggle/input/vicuna-7b-v1.5  # Đặt mô hình Vicuna tại thư mục này
train_data_path=/kaggle/input/librispeech/librispeech_train_960h.jsonl
val_data_path=/kaggle/input/librispeech/librispeech_dev_other.jsonl

output_dir=/kaggle/working/output/vicuna-7b-v1.5-WavLM-Large-libri960-ft-char-hotwords-$(date +"%Y%m%d")

# # Thiết lập Hydra arguments
# hydra_args="
# hydra.run.dir=$output_dir \
# ++model_config.llm_name=vicuna-7b-v1.5 \
# ++model_config.llm_path=$llm_path \
# ++model_config.llm_dim=4096 \
# ++model_config.encoder_name=wavlm \
# ++model_config.normalize=true \
# ++dataset_config.normalize=true \
# ++model_config.encoder_projector_ds_rate=5 \
# ++model_config.encoder_path=$speech_encoder_path \
# ++model_config.encoder_dim=1024 \
# ++model_config.encoder_projector=cov1d-linear \
# ++dataset_config.dataset=speech_dataset \
# ++dataset_config.train_data_path=$train_data_path \
# ++dataset_config.val_data_path=$val_data_path \
# ++dataset_config.input_type=raw \
# ++dataset_config.dataset=hotwords_dataset \
# ++dataset_config.file=examples/contextual_asr/dataset/hotwords_dataset.py:get_speech_dataset \
# ++train_config.model_name=asr \
# ++train_config.num_epochs=5 \
# ++train_config.freeze_encoder=true \
# ++train_config.freeze_llm=true \
# ++train_config.batching_strategy=custom \
# ++train_config.warmup_steps=1000 \
# ++train_config.total_steps=100000 \
# ++train_config.lr=1e-4 \
# ++train_config.validation_interval=8000 \
# ++train_config.val_batch_size=4 \
# ++train_config.batch_size_training=4 \
# ++train_config.num_workers_dataloader=2 \
# ++train_config.output_dir=$output_dir \
# ++metric=acc \
# ++log_config.log_file=$output_dir/train.log \
# ++log_config.use_wandb=false \  # Tắt wandb nếu không cần
# ++log_config.log_interval=5 \
# "

# # Huấn luyện với 2 GPU
# torchrun \
#     --nnodes=1 \  # Số node là 1 vì chạy trên một máy duy nhất
#     --nproc_per_node=2 \  # Số GPU sẽ sử dụng
#     --master_port=29504 \  # Cổng giao tiếp
#     $code_dir/finetune_contextual_asr.py \
#     --config-path "conf" \
#     --config-name "prompt.yaml" \
#     ++train_config.enable_fsdp=false \
#     ++train_config.enable_ddp=true \
#     ++train_config.use_fp16=true \
#     $hydra_args
