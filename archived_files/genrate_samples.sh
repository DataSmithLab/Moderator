example_gen_num=200
log_file="runtime_log.txt"
# 

Stable_Diffusion_Unet_Path="../stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin"
Pretrained_Unet_Path="../unet_backup/unet_original_diffusion_pytorch_model.bin"

Stable_Diffusion_CLIP_Path="../stable-diffusion-v1-5/text_encoder/pytorch_model.bin"
Pretrained_CLIP_Path="../unet_backup/encoder_orginal_pytorch_model.bin"

INIT_EXAMPLE_DATA_DIR="../LLMEthicsPatches/ziyao/data/big_gambling/train/0/"
mkdir $INIT_EXAMPLE_DATA_DIR

rm -rf $Stable_Diffusion_Unet_Path
ln -s $Pretrained_Unet_Path $Stable_Diffusion_Unet_Path
    
rm -rf $Stable_Diffusion_CLIP_Path
ln -s $Pretrained_CLIP_Path $Stable_Diffusion_CLIP_Path

python script/main_img_generate.py --saved_folder=$INIT_EXAMPLE_DATA_DIR --prompt="Bets" --gen_num=$example_gen_num

#for example_prompt in ${example_prompt_list[@]}
#do
#python script/img_generate.py --saved_folder=$INIT_EXAMPLE_DATA_DIR --prompt=$example_prompt --gen_num=$example_gen_num
#done;