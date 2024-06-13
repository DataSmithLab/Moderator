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
from lib.utils_task_vector import get_task_vector, task_vector_apply, save_task_vector, finetuned_unet_extract_safetensors
from lib.utils_data import image_compose, make_folder, dataset_make, generate_demo_imgs, generate_input_imgs_multi_prompts, SD_pipe
from lib.utils_merge import merge_main
import toml
from PIL import Image
from lib.edit_database import EditDatabase
import psutil

def get_database():
    edit_database = EditDatabase()
    return edit_database

def check_input_dataset_status(task_vector:dict):
    print("check_input_dataset_status", task_vector['input_data_dir'])
    if os.path.exists(task_vector['input_data_dir']):
        png_count = 0
        for file in os.listdir(task_vector['input_data_dir']):
            if file.endswith(".png"):
                png_count += 1
        if png_count >= task_vector["input_num"]:
            return 1
        else:
            return 0
    else:
        return 0

def check_trained_already(task_vector:dict, model_name:str="xl"):
    if model_name=="xl":
        if os.path.exists(task_vector['finetuned_model_dir']+"/output_model.safetensors") and os.path.exists(task_vector['finetuned_model_dir']+"/finetuned_unet.safetensors"):
            return 1
        else:
            return 0
    else:
        assert model_name=="xl"
    
def check_task_vector_saved(task_vector:dict, model_name:str="xl"):
    if model_name=="xl":
        if os.path.exists(task_vector['full_task_vector_path']):
            return 1
        else:
            return 0
    else:
        assert model_name=="xl"
        
#@app.route('/edit_status', method=['POST'])
def edit_status_check(config_data, args):
    
    config_data['gen_img_num_per_prompt']=1
    #config_data = request.get_data()
    #config_data = json.loads(config_data)
    model_name = config_data['model_name']
    #args = conceptPermissionConfig(model_name)
    
    task_vectors = config_data['task_vectors']
    edited_unet_path, whole_task_name=init_task_vector(task_vectors, args)
    
    for i in range(len(task_vectors)):
        config_data['task_vectors'][i]['gen_img_num_per_prompt']=1
        config_data['task_vectors'][i]['input_data_init']=check_input_dataset_status(config_data['task_vectors'][i])
        config_data['task_vectors'][i]['trained']=check_trained_already(config_data['task_vectors'][i], model_name)
        config_data['task_vectors'][i]['saved'] = check_task_vector_saved(config_data['task_vectors'][i], model_name)
    return config_data
    #edited_unet_path = model_edit()

def resize_image(input_path, output_path):
    with Image.open(input_path) as img:
        # 修改图片尺寸为512x512
        resized_img = img.resize((512, 512))

        # 保存修改后的图片
        resized_img.save(output_path, format='PNG')
    return f"Image resized and saved to {output_path}"

def convert_to_dict(input_str):
    input_str = input_str.replace("{", "")
    input_str = input_str.replace("}", "")
    input_str = input_str.replace("\n", "")
    #input_str = input_str.replace('"', "")
    # 将字符串分割成键值对
    pairs = input_str.split('", "')

    # 初始化空字典
    result_dict = {}

    # 遍历每个键值对
    for pair in pairs:
        # 分割键和值
        ##print(pair)
        key, value = pair.split('": "')

        # 去除多余的字符并将键值对添加到字典中
        result_dict[key.strip("\"")] = value.strip("\"")

    return result_dict

def dataset_adjust(input_data_dir):
    f = open(input_data_dir+"/metadata.jsonl", "r")
    meta_data = f.readlines()
    new_dict = {}
    for meta_data_line in meta_data:
        meta_data_dict = convert_to_dict(meta_data_line)
        #print(meta_data_dict)
        img_key = meta_data_dict["file_name"]
        new_img_key = img_key.replace(" ","_")
        img_caption = meta_data_dict["text"]
        if os.path.exists(input_data_dir+"/"+new_img_key):
            pass
        else:
            os.rename(input_data_dir+"/"+img_key, input_data_dir+"/"+new_img_key)
        
        print(input_data_dir+"/"+new_img_key)
        resize_image(input_data_dir+"/"+new_img_key, input_data_dir+"/"+new_img_key)
        
        new_dict[new_img_key] = {'caption':img_caption, 'tags':"violence"}
    with open(input_data_dir+"/metadata.json", "w+") as f:
        json.dump(new_dict, f)

def xl_finetune_toml_make(task_vector:dict):
    toml_path = task_vector['finetuned_model_dir']+"/fine_tune.toml"
    
    dataset_adjust(task_vector['input_data_dir'])
    
    toml_config = {
        'general': {
            'shuffle_caption': True, 
            'keep_tokens': 1
        },
        'datasets': [
            {
                'resolution': [512, 512],
                'batch_size': 1,
                'subsets': [
                    {
                        'image_dir': task_vector['input_data_dir'],
                        'metadata_file': task_vector['input_data_dir']+'/metadata.json'
                    }
                ]
            }
        ]
    }
    with open(toml_path, 'w+') as f:
        toml.dump(toml_config, f)
    return toml_path

def finetuned_unet_extract_xl(finetuned_model_name, finetuned_unet_name, backup_unet_name):
    device="cpu"
    _, state_dict = load_checkpoint_with_text_encoder_conversion(finetuned_model_name, device=device)
    unet_use_linear_projection_in_v2=True
    v2=True
    unet_config = create_unet_diffusers_config(v2, unet_use_linear_projection_in_v2)
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(v2, state_dict, unet_config)
    backup_unet_state_dict = load_file(backup_unet_name)
    converted_unet_checkpoint['add_embedding.linear_1.bias'] = backup_unet_state_dict['add_embedding.linear_1.bias']
    converted_unet_checkpoint['add_embedding.linear_2.bias'] = backup_unet_state_dict['add_embedding.linear_2.bias']
    converted_unet_checkpoint['add_embedding.linear_2.weight'] = backup_unet_state_dict['add_embedding.linear_2.weight']
    converted_unet_checkpoint['add_embedding.linear_1.weight'] = backup_unet_state_dict['add_embedding.linear_1.weight']
    save_file(converted_unet_checkpoint, finetuned_unet_name)

def finetune_model_xl(task_vector:dict, model_id):
    make_folder(task_vector['finetuned_model_dir'])
    toml_path=xl_finetune_toml_make(task_vector)
    script_path = "/home/featurize/work/sd-scripts/sdxl_train.py"
    args = [
        "--pretrained_model_name_or_path", model_id,
        "--output_dir", task_vector['finetuned_model_dir'],
        "--output_name", "output_model",
        "--dataset_config", toml_path,
        "--save_model_as", "safetensors",
        "--learning_rate", "1e-6",
        "--max_train_steps", str(task_vector['train_step']),
        "--use_8bit_adam", "--xformers",
        "--gradient_checkpointing",
        "--mixed_precision", "fp16",
        "--cache_latents", "--no_half_vae"
    ]
    command = ["python", script_path] + args
    command_str = " ".join(args)
    print(command_str)
    subprocess.run(command, check=True)
    finetuned_unet_extract_safetensors(task_vector['finetuned_model_dir']+"/output_model.safetensors", task_vector['finetuned_model_dir']+"/finetuned_unet.safetensors", "/home/featurize/work/stable-diffusion-xl-base-1.0/unet_backup/diffusion_pytorch_model.safetensors")

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
    torch.cuda.empty_cache()
    print('after finetune_model', torch.cuda.memory_summary())

def finetune_on_task(task_vector, args, model_name="1.5"):
    
    print(task_vector['input_data_init'])
    Stable_Diffusion_Unet_Path=args.sd_unet_path
    Pretrained_Unet_Path=args.pretrain_unet_path
    model_id = args.sd_path
    
    if task_vector['trained'] or os.path.exists(task_vector['finetuned_unet_path']):
        print(task_vector['finetuned_unet_path'], " trained already")
    else:
        if task_vector['input_data_init']==0:
            img_generate_start_time = time.time()
            generate_input_imgs_multi_prompts(task_vector, Stable_Diffusion_Unet_Path, Pretrained_Unet_Path, args.sd_path, model_name)
            torch.cuda.empty_cache()
            print('after generate_input_imgs_multi_prompts', torch.cuda.memory_summary())
            img_generate_end_time = time.time()
            img_generate_time = img_generate_end_time-img_generate_start_time
            print('finetune image generate time', img_generate_time)
        else:
            print("input data ready")
            
        time.sleep(60)
        
        finetune_start_time = time.time()
        if model_name=="1.5":
            finetune_model(task_vector, model_id)
        elif model_name == "xl":
            finetune_model_xl(task_vector, model_id)
        finetune_end_time = time.time()
        finetune_time = finetune_end_time - finetune_start_time
        print('finetune model time', finetune_time)
    save_task_vector(Pretrained_Unet_Path, task_vector, model_name)
    
def plot_imgs(unet_path, example_gen_num, example_prompts, example_names, before_folder_name, gen_img_num_per_prompt, args, width=1024, height=1024, model_name="1.5"):
    Stable_Diffusion_Unet_Path=args.sd_unet_path
    if model_name == "xl":
        model_id = args.sdxl_path
    elif model_name == "1.5":
        model_id = args.sd_path
    fresh_sd(Stable_Diffusion_Unet_Path, unet_path)
    if os.path.exists(before_folder_name):
        pass
    else:
        make_folder(before_folder_name)
    image_names = generate_demo_imgs(
        model_id, 
        example_gen_num, 
        example_prompts, 
        example_names, 
        before_folder_name, 
        gen_img_num_per_prompt, 
        width=width, 
        height=height, 
        model_name=model_name
    )
    torch.cuda.empty_cache()
    print('after plot_imgs', torch.cuda.memory_summary())
    return image_names

    
def model_edit(config_data, args, model_name):
    
    Stable_Diffusion_Unet_Path=args.sd_unet_path
    Pretrained_Unet_Path=args.pretrain_unet_path
    model_id = args.sd_path
    
    task_vector_applied = config_data['task_vector_applied']
    task_vectors = config_data['task_vectors']
    merge = config_data['merge']
    
    edited_unet_path, whole_task_name=init_task_vector(task_vectors, args)
    if 'edited_unet_path' in config_data:
        edited_unet_path = config_data['edited_unet_path']
    
    fresh_sd(Stable_Diffusion_Unet_Path, Pretrained_Unet_Path)

    for task_vector in task_vectors:
        finetune_on_task(task_vector, args, model_name)
    if task_vector_applied & os.path.exists(edited_unet_path):
        pass
    else:
        task_vector_apply(Pretrained_Unet_Path, edited_unet_path, task_vectors, merge, model_name)
    return edited_unet_path


def model_pipeline(unet_path, model_name, args):
    Stable_Diffusion_Unet_Path=args.sd_unet_path
    if model_name == "xl":
        model_id = args.sdxl_path
    elif model_name == "1.5":
        model_id = args.sd_path
    fresh_sd(Stable_Diffusion_Unet_Path, unet_path)
    return SD_pipe(args.sdxl_path, model_name)