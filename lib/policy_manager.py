import json
import flask
from flask import Flask, jsonify, request, abort
import yaml
from lib.utils_serve import model_edit, plot_imgs, edit_status_check, get_database
import os
import argparse
from PIL import Image
from lib.utils_config import exp_config_gen

class conceptPermissionConfig:
    def __init__(self, model_name="1.5"):
        self.model_name = model_name
        self.work_dir = "/home/featurize/work/ModeratorAE"#os.environ.get("LLMEthicsPatchHome")
        if self.model_name == "1.5":
            self.sd_path = self.work_dir+"/stable-diffusion-v1-5"
            self.sd_unet_path=self.work_dir+"/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin"
            self.pretrain_unet_path=self.work_dir+"/unet_backup/unet_original_diffusion_pytorch_model.bin"
        elif self.model_name == "xl":
            self.sdxl_path = self.work_dir+"/stable-diffusion-xl-base-1.0"
            self.sd_path = self.work_dir+"/stable-diffusion-xl-base-1.0"
            self.sd_unet_path=self.work_dir+"/stable-diffusion-xl-base-1.0/unet/diffusion_pytorch_model.safetensors"
            self.pretrain_unet_path=self.work_dir+"/stable-diffusion-xl-base-1.0/unet_backup/diffusion_pytorch_model.safetensors"
        #self.data_dir=self.work_dir+"/LLMEthicsPatches/data"
        #self.finetuned_models_dir=self.work_dir+"/LLMEthicsPatches/files/models_finetune/"
        #self.edited_models_dir=self.work_dir+"/LLMEthicsPatches/files/models_edited/"
        #self.task_vectors_dir=self.work_dir+"/LLMEthicsPatches/files/task_vectors/"      
        self.data_dir=self.work_dir+"/ConceptPermission/data"
        self.finetuned_models_dir=self.work_dir+"/ConceptPermission/files/models_finetune/"
        self.edited_models_dir=self.work_dir+"/ConceptPermission/files/models_edited/"
        self.task_vectors_dir=self.work_dir+"/ConceptPermission/files/task_vectors/"

class ModeratorPolicyManager:
    def __init__(self):
        self.database_path = "database/policy_database.json"
        self.policy_database = self.load_database()

    def load_database(self):
        with open(self.database_path, "r")as f:
            policy_database = json.load(f)
        return policy_database

    def write_database(self):
        with open(self.database_path, "w")as f:
            json.dump(self.policy_database, f)

    def add_policy(self, new_policy_dict, new_policy_name):
        self.policy_database[new_policy_name] = new_policy_dict
        self.write_database()

    def single_task_vector_parse(
            self,
            label_content_dict:dict, # {obj:, sty:, act:}
            real_content_dict:dict,
            content_operator:str, # "-" or "+"
            expand_type,
            expand_key
    ):
        content_name = real_content_dict["obj"].replace(" ", "_")+"-"+real_content_dict["sty"].replace(" ", "_")+"-"+real_content_dict["act"].replace(" ", "_")
        task_vector_dict = {
            "input_data_init": 0,
            "input_num": 100,
            "name": content_name,
            "operator": content_operator,
            "images_configs":{
                "image_name": content_name,
                "expand_type": expand_type,
                "label_context": {
                    "obj": label_content_dict["obj"],
                    "sty": label_content_dict["sty"],
                    "act": label_content_dict["act"],
                },
                "real_context": {
                    "obj": real_content_dict["obj"],
                    "sty": real_content_dict["sty"],
                    "act": real_content_dict["act"],
                },
                "expand_key": expand_key
            }
        }
        return task_vector_dict

    def policy_parse_to_yaml(self, policy_dict):
        yaml_dict = {
            "task_vector_applied": 0,
            "merge": False,
            "model_name": "xl",
            "task_vectors": []
        }
        if policy_dict["method"]=="REMOVE":
            remove_task_vector = self.single_task_vector_parse(
                label_content_dict=policy_dict["src_content"],
                real_content_dict=policy_dict["src_content"],
                content_operator = "-",
                expand_type = policy_dict["expand_type"],
                expand_key = policy_dict["expand_context"]
            )
            yaml_dict["task_vectors"].append(remove_task_vector)
        elif policy_dict["method"]=="REPLACE":
            src_task_vector = self.single_task_vector_parse(
                label_content_dict=policy_dict["src_content"],
                real_content_dict=policy_dict["src_content"],
                content_operator="-",
                expand_type=policy_dict["expand_type"],
                expand_key=policy_dict["expand_context"]
            )
            dst_task_vector = self.single_task_vector_parse(
                label_content_dict=policy_dict["dst_content"],
                real_content_dict=policy_dict["src_content"],
                content_operator="+",
                expand_type=policy_dict["expand_type"],
                expand_key=policy_dict["expand_context"]
            )
            yaml_dict["task_vectors"].append(src_task_vector)
            yaml_dict["task_vectors"].append(dst_task_vector)
        elif policy_dict["method"]=="MOSAIC":
            src_task_vector = self.single_task_vector_parse(
                label_content_dict=policy_dict["src_content"],
                real_content_dict=policy_dict["src_content"],
                content_operator="-",
                expand_type=policy_dict["expand_type"],
                expand_key=policy_dict["expand_context"]
            )
            dst_task_vector = self.single_task_vector_parse(
                label_content_dict=policy_dict["src_content"],
                real_content_dict=policy_dict["src_content"],
                content_operator="+",
                expand_type=policy_dict["expand_type"],
                expand_key=policy_dict["expand_context"]
            )
            yaml_dict["task_vectors"].append(src_task_vector)
            yaml_dict["task_vectors"].append(dst_task_vector)
        return yaml_dict


    def craft_policy(
            self,
            policy_dict,
            policy_name
    ):
        config_yaml = self.policy_parse_to_yaml(
            policy_dict=policy_dict
        )
        with open("tmp_config_edit.yaml", "w+") as f:
            yaml.dump(config_yaml, f)
        edited_unet_path = self.call_edit_backend(
            config_yaml="tmp_config_edit.yaml"
        )
        policy_dict["edited_model_path"]=edited_unet_path
        self.add_policy(
            new_policy_dict=policy_dict,
            new_policy_name=policy_name
        )
        return "Success"

    def get_policy(
            self,
            policy_name
    ):
        return self.policy_database[policy_name]

    def call_policy_model(
            self,
            policy_name,
            prompt
    ):
        policy_dict = self.get_policy(
            policy_name=policy_name
        )
        unet_path = policy_dict["edited_model_path"]
        img_yaml = {
            "folder_name": policy_name,
            "gen_img_num_per_prompt": 1,
            "img_gen_num": 9,
            "img_names": [policy_name],
            "img_prompts": [prompt],
            "model_name": "xl",
            "unet_path":unet_path
        }
        with open("tmp_config_generate.yaml", "w+") as f:
            yaml.dump(img_yaml, f)
        image_name_list_str = self.call_edit_generate_backend(
            config_yaml="tmp_config_generate.yaml"
        )
        image_name_list = eval(image_name_list_str)
        return image_name_list

    def call_pretrain_model(
            self,
            prompt
    ):
        config_data = {
            "folder_name": "test",
            "gen_img_num_per_prompt": 1,
            "img_gen_num": 9,
            "img_names": ["test"],
            "img_prompts": [prompt],
            "model_name": "xl",
        }
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
        return image_names

    def call_edit_backend(
            self,
            config_yaml
    ):
        with open(config_yaml, "r") as yaml_file:
            config_data = yaml.safe_load(yaml_file)
        data = json.dumps(config_data)
        
        model_name = data['model_name']
        args = conceptPermissionConfig(model_name)
        config_data = edit_status_check(data, args)
        edited_unet_path = model_edit(data, args, model_name)
        return edited_unet_path

    def call_edit_generate_backend(
            self,
            config_yaml
    ):
        url = 'http://127.0.0.1:5000/img_generate'
        # config_yaml="config_img.yaml')"
        with open(config_yaml, "r") as yaml_file:
            config_data = yaml.safe_load(yaml_file)
        
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
        return image_names