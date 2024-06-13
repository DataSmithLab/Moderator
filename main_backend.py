from flask import Flask, jsonify, request, abort
import json
app = Flask(__name__)
from lib.utils_serve import model_edit, plot_imgs, edit_status_check, get_database
import os
import argparse
from PIL import Image
from lib.utils_config import exp_config_gen

class conceptPermissionConfig:
    def __init__(self, model_name="1.5"):
        self.model_name = model_name
        self.work_dir = "/home/featurize/work"#os.environ.get("LLMEthicsPatchHome")
        if self.model_name == "1.5":
            self.sd_path = self.work_dir+"/stable-diffusion-v1-5"
            self.sd_unet_path=self.work_dir+"/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin"
            self.pretrain_unet_path=self.work_dir+"/unet_backup/unet_original_diffusion_pytorch_model.bin"
        elif self.model_name == "xl":
            self.sdxl_path = self.work_dir+"/stable-diffusion-xl-base-1.0"
            self.sd_path = self.work_dir+"/stable-diffusion-xl-base-1.0"
            self.sd_unet_path=self.work_dir+"/stable-diffusion-xl-base-1.0/unet/diffusion_pytorch_model.safetensors"
            self.pretrain_unet_path=self.work_dir+"/stable-diffusion-xl-base-1.0/unet_backup/diffusion_pytorch_model.safetensors"
        self.data_dir=self.work_dir+"/LLMEthicsPatches/data"
        self.finetuned_models_dir=self.work_dir+"/LLMEthicsPatches/files/models_finetune/"
        self.edited_models_dir=self.work_dir+"/LLMEthicsPatches/files/models_edited/"
        self.task_vectors_dir=self.work_dir+"/LLMEthicsPatches/files/task_vectors/"

@app.route('/pretrain_img_generate', methods=['POST'])
def pretrain_img_generate():
    config_data = request.get_data()
    config_data =  json.loads(config_data)
    
    #unet_path = config_data['unet_path']
    example_gen_num = config_data['img_gen_num']
    example_prompts = config_data['img_prompts']
    example_names = config_data['img_names']
    model_name = config_data['model_name']
    
    args = conceptPermissionConfig(model_name)
    unet_path = args.pretrain_unet_path
    folder_name = args.data_dir+config_data['folder_name']
    
    gen_img_num_per_prompt = config_data['gen_img_num_per_prompt']
    gen_img_num_per_prompt = 1 # set default to 1 to enable minimal memory consumption, you can delete this line of code
    
    width=1024
    height=1024
    
    image_names = plot_imgs(unet_path, example_gen_num, example_prompts, example_names, folder_name, gen_img_num_per_prompt, args, width=width, height=height, model_name=model_name)
    return str(image_names)
        
@app.route('/query_database', methods=['GET'])
def query_database():
    return get_database()

@app.route('/craft_config', methods=['POST'])
def craft_config():
    policy_data = request.get_data()
    policy_data = json.loads(policy_data)
    config_filename=exp_config_gen(        
        src_content=policy_data["src_content"],
        dst_content=policy_data["dst_content"],
        src_name=policy_data["src_name"],
        dst_name=policy_data["dst_name"],
        task_name=policy_data["task_name"],
        plot_img_contents=[],
        method=policy_data["method"]
    )
    return config_filename
    
        
@app.route('/model_edit', methods=['POST'])
def edit():
    
    config_data = request.get_data()
    config_data =  json.loads(config_data)
    
    #database=get_database()
    #database.add_task(config_data)
    #database.store_database()
    model_name = config_data['model_name']
    args = conceptPermissionConfig(model_name)
    config_data = edit_status_check(config_data, args)
    edited_unet_path = model_edit(config_data, args, model_name)
    return edited_unet_path

@app.route('/img_generate', methods=['POST'])
def img_generate():
    config_data = request.get_data()
    config_data =  json.loads(config_data)
    
    unet_path = config_data['unet_path']
    example_gen_num = config_data['img_gen_num']
    example_prompts = config_data['img_prompts']
    example_names = config_data['img_names']
    model_name = config_data['model_name']
    args = conceptPermissionConfig(model_name)
    folder_name = args.data_dir+config_data['folder_name']
    
    gen_img_num_per_prompt = config_data['gen_img_num_per_prompt']
    gen_img_num_per_prompt = 1 # set default to 1 to enable minimal memory consumption, you can delete this line of code
    
    width=2048
    height=2048
    
    image_names = plot_imgs(unet_path, example_gen_num, example_prompts, example_names, folder_name, gen_img_num_per_prompt, args, width=width, height=height, model_name=model_name)
    return str(image_names)