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
from lib.utils_task_vector import get_task_vector, task_vector_apply, save_task_vector
from lib.utils_data import image_compose, make_folder, dataset_make, generate_demo_imgs, generate_input_imgs, generate_input_imgs_multi_prompts
from lib.utils_merge import merge_main

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

def finetune_on_task(task_vector, args):
    
    Stable_Diffusion_Unet_Path=args.sd_unet_path
    Pretrained_Unet_Path=args.pretrain_unet_path
    model_id = args.sd_path
    
    if task_vector['trained']&os.path.exists(task_vector['finetuned_unet_path']):
        print(task_vector['finetuned_unet_path'], " trained already")
    else:
        if task_vector['input_data_init']==0:
            img_generate_start_time = time.time()
            generate_input_imgs_multi_prompts(task_vector, Stable_Diffusion_Unet_Path, Pretrained_Unet_Path, args.sd_path)
            img_generate_end_time = time.time()
            img_generate_time = img_generate_end_time-img_generate_start_time
            print('finetune image generate time', img_generate_time)
        else:
            print("input data ready")
        finetune_start_time = time.time()
        finetune_model(task_vector, model_id)
        finetune_end_time = time.time()
        finetune_time = finetune_end_time - finetune_start_time
        print('finetune model time', finetune_time)
    save_task_vector(Pretrained_Unet_Path, task_vector)
    
def plot_imgs(unet_path, example_gen_num, example_prompts, example_names, before_folder_name, gen_img_num_per_prompt, args):
    Stable_Diffusion_Unet_Path=args.sd_unet_path
    model_id = args.sd_path
    fresh_sd(Stable_Diffusion_Unet_Path, unet_path)
    image_names = generate_demo_imgs(model_id, example_gen_num, example_prompts, example_names, before_folder_name, gen_img_num_per_prompt)
    return image_names

    
def model_edit(config_data, args):
    
    Stable_Diffusion_Unet_Path=args.sd_unet_path
    Pretrained_Unet_Path=args.pretrain_unet_path
    model_id = args.sd_path
    
    task_vector_applied = config_data['task_vector_applied']
    task_vectors = config_data['task_vectors']
    merge = config_data['merge']
    
    edited_unet_path, whole_task_name=init_task_vector(task_vectors, args)
    
    fresh_sd(Stable_Diffusion_Unet_Path, Pretrained_Unet_Path)

    for task_vector in task_vectors:
        finetune_on_task(task_vector, args)
    if task_vector_applied & os.path.exists(edited_unet_path):
        pass
    else:
        task_vector_apply(Pretrained_Unet_Path, edited_unet_path, task_vectors, merge)
    return edited_unet_path