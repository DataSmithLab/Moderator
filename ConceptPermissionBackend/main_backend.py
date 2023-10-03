from flask import Flask, jsonify, request, abort
import json
app = Flask(__name__)
from lib.utils_serve import model_edit, plot_imgs
import os
import argparse

class conceptPermissionConfig:
    work_dir = os.environ.get("LLMEthicsPatchHome")
    sd_path = work_dir+"/stable-diffusion-v1-5"
    sd_unet_path=work_dir+"/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin"
    pretrain_unet_path=work_dir+"/unet_backup/unet_original_diffusion_pytorch_model.bin"
    data_dir=work_dir+"/LLMEthicsPatches/data"
    finetuned_models_dir=work_dir+"/LLMEthicsPatches/models_finetune/"
    edited_models_dir=work_dir+"/LLMEthicsPatches/models_edited/"
    task_vectors_dir=work_dir+"/LLMEthicsPatches/task_vectors/"
    
args = conceptPermissionConfig()


@app.route('/model_edit', methods=['POST'])
def edit():
    config_data = request.get_data()
    config_data =  json.loads(config_data)
    edited_unet_path = model_edit(config_data, args)
    return edited_unet_path

@app.route('/img_generate', methods=['POST'])
def img_generate():
    config_data = request.get_data()
    config_data =  json.loads(config_data)
    
    unet_path = config_data['unet_path']
    example_gen_num = config_data['img_gen_num']
    example_prompts = config_data['img_prompts']
    example_names = config_data['img_names']
    folder_name = args.data_dir+config_data['folder_name']
    
    gen_img_num_per_prompt = config_data['gen_img_num_per_prompt']
    gen_img_num_per_prompt = 1 # set default to 1 to enable minimal memory consumption, you can delete this line of code
    
    image_names = plot_imgs(unet_path, example_gen_num, example_prompts, example_names, folder_name, gen_img_num_per_prompt, args)
    return str(image_names)