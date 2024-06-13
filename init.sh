mkdir data
mkdir summary
mkdir edited_models
mkdir finetuned_models
mkdir generated_data

cd ..
mkdir unet_backup
mv $LLMEthicsPatchHome/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin $LLMEthicsPatchHome/unet_backup/unet_original_diffusion_pytorch_model.bin
mv $LLMEthicsPatchHome/stable-diffusion-v1-5/text_encoder/pytorch_model.bin $LLMEthicsPatchHome/unet_backup/encoder_orginal_pytorch_model.bin