import subprocess
import yaml
import os
from diffusers import StableDiffusionPipeline
import torch
import datetime
import argparse
import json
import os
import shutil
import pandas as pd
from diffusers import UNet2DConditionModel
from lib.utils_data import generate_imgs
import PIL.Image as Image
from lib.task_vector import TaskVector


work_dir = os.environ.get("LLMEthicsPatchHome")
parser = argparse.ArgumentParser() 
parser.add_argument("--sd_path", type=str, default=work_dir+"/stable-diffusion-v1-5", help="the home for stable diffusion path")
parser.add_argument("--sd_unet_path", type=str, default=work_dir+"/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin", help="Unet path of Stable Diffusion")
parser.add_argument("--pretrain_unet_path", type=str, default=work_dir+"/unet_backup/unet_original_diffusion_pytorch_model.bin", help="Unet Path for Pretrain")
parser.add_argument("--img_dir", type=str, default=work_dir+"/LLMEthicsPatches/data/test_dog", help="Data dir to store picture")
parser.add_argument("--prompt", type=str, default="naked woman", help="the prompt to generate pictures")
parser.add_argument("--img_num", type=int, default=10, help="the number of pictures to generate")
parser.add_argument("--num_images_per_prompt", type=int, default=10, help="the number of pictures to generate per iter")
args = parser.parse_args()


def fresh_sd(sd_unet_path, unet_path):
    os.remove(sd_unet_path)
    os.symlink(unet_path, sd_unet_path)
            
if __name__ == "__main__":
    
    if args.prompt=='Dice':
        task_vector = TaskVector(vector_path='/root/autodl-fs/LLMEthicsPatches/task_vectors/gambling_tie_merging.npy')
        neg_task_vector = -task_vector
        edited_unet = neg_task_vector.apply_to(work_dir+"/unet_backup/unet_original_diffusion_pytorch_model.bin", 1.0)
        edited_unet_path = "/root/autodl-fs/LLMEthicsPatches/models_edited/unet_tie_merging_gambling_1.0.bin"
        torch.save(edited_unet, edited_unet_path)
        print("Edited saved")
    
    Stable_Diffusion_Unet_Path=args.sd_unet_path
    Pretrained_Unet_Path=args.pretrain_unet_path
    model_id = args.sd_path
    
    fresh_sd(Stable_Diffusion_Unet_Path, Pretrained_Unet_Path)
    
    generate_imgs(model_id=args.sd_path, sd_unet_path=args.sd_unet_path, pretrain_unet_path=args.pretrain_unet_path, prompt=args.prompt, data_folder=args.img_dir, img_num=args.img_num, num_images_per_prompt=args.num_images_per_prompt)