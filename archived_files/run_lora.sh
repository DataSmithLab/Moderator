rm -rf /root/autodl-fs/stable-diffusion-v1-5-backend/unet/diffusion_pytorch_model.bin
ln -s /root/autodl-fs/unet_backup/unet_original_diffusion_pytorch_model.bin /root/autodl-fs/stable-diffusion-v1-5-backend/unet/diffusion_pytorch_model.bin


python /root/autodl-fs/stable-diffusion/train_text_to_image_LoRa.py \
    --mixed_precision="fp16" \
  --pretrained_model_name_or_path=/root/autodl-fs/stable-diffusion-v1-5 \
  --dataset_name="/root/autodl-fs/LLMEthicsPatches/data/input-lora-test" --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=30 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="/root/autodl-fs/LLMEthicsPatches/models_finetune/finetuned_Unet-lora-gun-100" \
  --validation_prompt="gun"
  

python lora_test.py