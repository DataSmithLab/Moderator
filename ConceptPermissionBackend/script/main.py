import subprocess
import yaml
import os
import torch
import datetime
import argparse
import json
import os
import shutil
import pandas as pd
import sys
from diffusers import UNet2DConditionModel
import time
from lib.utils import fresh_sd, init_task_vector
from lib.utils_task_vector import get_task_vector, task_vector_apply
from lib.utils_data import image_compose, make_folder, dataset_make, generate_demo_imgs, generate_input_imgs, generate_input_imgs_multi_prompts

work_dir = os.environ.get("LLMEthicsPatchHome")
parser = argparse.ArgumentParser() 
parser.add_argument("--sd_path", type=str, default=work_dir+"/stable-diffusion-v1-5", help="the home for stable diffusion path")
parser.add_argument("--sd_unet_path", type=str, default=work_dir+"/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin", help="Unet path of Stable Diffusion")
parser.add_argument("--pretrain_unet_path", type=str, default=work_dir+"/unet_backup/unet_original_diffusion_pytorch_model.bin", help="Unet Path for Pretrain")
parser.add_argument("--data_dir", type=str, default=work_dir+"/LLMEthicsPatches/data", help="Data dir")
parser.add_argument("--finetuned_models_dir", type=str, default=work_dir+"/LLMEthicsPatches/models_finetune/", help="Models dir")
parser.add_argument("--edited_models_dir", type=str, default=work_dir+"/LLMEthicsPatches/models_edited/", help="Models dir")
parser.add_argument("--task_vectors_dir", type=str, default=work_dir+"/LLMEthicsPatches/task_vectors/", help="Task Vectors dir")
parser.add_argument("--config_yaml", type=str, default="config.yaml", help="the config file")
args = parser.parse_args()

def finetune_model(task_vector:dict, model_name):
    make_folder(task_vector['finetuned_model_dir'])
    script_path = "lib/train_text_to_image.py"
    args = [
            "--pretrained_model_name_or_path", model_name,
            "--train_data_dir", task_vector['input_data_dir'],
            "--resolution", "512",
            "--center_crop",
            "--random_flip",
            "--train_batch_size", "1",
            "--use_ema",
            "--gradient_accumulation_steps", "1",
            "--gradient_checkpointing",
            "--mixed_precision", "no",
            "--max_train_steps", str(task_vector['train_step']),
            "--learning_rate", "1e-05",
            "--max_grad_norm", "1",
            "--lr_scheduler", "constant",
            "--lr_warmup_steps", "0",
            "--output_dir",  task_vector['finetuned_model_dir'],
            "--validation_prompt", task_vector['name'],
            "--enable_xformers_memory_efficient_attention",
            "--checkpointing_steps", "10000"
    ]
    command = ["python", script_path] + args
    command_str = " ".join(args)
    print(command_str)
    subprocess.run(command, check=True)

if __name__ == "__main__":
    
    with open(args.config_yaml, "r") as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    
    #log_filename = "logs/"+config_data['before_folder_name']+".log"
    #log_file = open(log_filename, 'w+')
    # 重定向标准输出和标准错误
    #sys.stdout = log_file
    #sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
    #sys.stderr = log_file
    
    Stable_Diffusion_Unet_Path=args.sd_unet_path
    Pretrained_Unet_Path=args.pretrain_unet_path
    model_id = args.sd_path
    
    task_vector_applied = config_data['task_vector_applied']
    task_vectors = config_data['task_vectors']
    example_prompts = config_data['example_prompts']
    example_names = config_data['example_names']
    example_gen_num = config_data['example_gen_num']
    gen_img_num_per_prompt=config_data['gen_img_num_per_prompt']
    
    before_folder_name = args.data_dir+config_data['before_folder_name']
    after_folder_name = args.data_dir+config_data['after_folder_name']
    
    plot_init_example = config_data['plot_init_example']
    
    example_group_config = config_data['example_group_config']
    image_column = example_group_config['image_column']
    image_row = example_group_config['image_row']
    image_size = example_group_config['image_size']
    
    edited_unet_path, whole_task_name=init_task_vector(task_vectors, args)
    #before_folder_name=args.data_dir+whole_task_name+"-before"
    
    
    fresh_sd(Stable_Diffusion_Unet_Path, Pretrained_Unet_Path)
    if plot_init_example:
        generate_demo_imgs(model_id, example_gen_num, example_prompts, example_names, before_folder_name, gen_img_num_per_prompt)
    for task_vector in task_vectors:
        if task_vector['trained']==0:
            #if 'input_data_init' not in task_vector:
            if task_vector['input_data_init']==0:
                img_generate_start_time = time.time()
                generate_input_imgs_multi_prompts(task_vector, Stable_Diffusion_Unet_Path, Pretrained_Unet_Path, args.sd_path)
                img_generate_end_time = time.time()
                img_generate_time = img_generate_end_time-img_generate_start_time
                print('image generate time', img_generate_time)
            finetune_start_time = time.time()
            finetune_model(task_vector, model_id)
            finetune_end_time = time.time()
            finetune_time = finetune_end_time - finetune_start_time
            print('finetune time', finetune_time)
    if task_vector_applied:
        pass
    else:
        task_vector_apply(Pretrained_Unet_Path, edited_unet_path, task_vectors)
    fresh_sd(Stable_Diffusion_Unet_Path, edited_unet_path)
    #folder_name=args.data_dir+whole_task_name+"-after"
    generate_demo_imgs(model_id, example_gen_num, example_prompts, example_names, after_folder_name, num_images_per_prompt=gen_img_num_per_prompt)