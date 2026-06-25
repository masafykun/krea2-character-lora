#!/bin/bash
cd /mnt/data/krea2-lora/musubi-tuner
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/mnt/data/krea2-lora/logs
M=/mnt/data/krea2-lora/models
DS=/mnt/data/krea2-lora/dataset.toml
OUT=/mnt/data/krea2-lora/output
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  src/musubi_tuner/krea2_train_network.py \
  --dit $M/krea2_raw.safetensors --vae $M/qwen_image_vae.safetensors --dataset_config $DS \
  --sdpa --mixed_precision bf16 --fp8_base --fp8_scaled --blocks_to_swap 26 \
  --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 2.5 \
  --network_module networks.lora_krea2 --network_dim 32 --network_alpha 32 \
  --optimizer_type adamw --learning_rate 1e-4 \
  --max_train_epochs 10 --save_every_n_epochs 2 --gradient_checkpointing --seed 42 \
  --output_dir $OUT --output_name masafee_krea2 --logging_dir $L
