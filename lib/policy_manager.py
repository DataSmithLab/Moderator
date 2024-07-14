import json
import flask
from flask import Flask, jsonify, request, abort
import yaml
import requests

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
        img_yaml = {
            "folder_name": "test",
            "gen_img_num_per_prompt": 1,
            "img_gen_num": 9,
            "img_names": ["test"],
            "img_prompts": [prompt],
            "model_name": "xl",
        }
        with open("tmp_config_generate.yaml", "w+") as f:
            yaml.dump(img_yaml, f)
        image_name_list_str = self.call_edit_generate_backend(
            config_yaml="tmp_config_generate.yaml"
        )
        image_name_list = eval(image_name_list_str)
        return image_name_list

    def call_edit_backend(
            self,
            config_yaml
    ):
        url = 'http://127.0.0.1:5000/model_edit'
        # config_yaml="config_edit.yaml')"
        with open(config_yaml, "r") as yaml_file:
            config_data = yaml.safe_load(yaml_file)
        data = json.dumps(config_data)
        r = requests.post(url, data=data, timeout=10000)
        while True:
            if r.status_code == 500:
                r = requests.post(url, data=data, timeout=10000)
            else:
                break
        return r.text

    def call_edit_generate_backend(
            self,
            config_yaml
    ):
        url = 'http://127.0.0.1:5000/img_generate'
        # config_yaml="config_img.yaml')"
        with open(config_yaml, "r") as yaml_file:
            config_data = yaml.safe_load(yaml_file)
        print(config_data)
        data = json.dumps(config_data)
        r = requests.post(url, data=data, timeout=100000)
        return r.text