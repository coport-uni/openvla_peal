torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /root/tensorflow_datasets \
  --dataset_name NeuromekaNet \
  --run_root_dir /workspace/OpenVLA/root_dir \
  --adapter_tmp_dir /workspace/OpenVLA/tmp_dir \
  --lora_rank 32 \
  --batch_size 2 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project openvla-lora \
  --wandb_entity ohsungwoo-unist \
  --save_steps 1000

# batch_size 3 not recommended! 