import requests
import json
import yaml


def edit_request_example(config_yaml):
    url = 'http://127.0.0.1:5000/model_edit'
    #config_yaml="config_edit.yaml')"
    with open(config_yaml, "r") as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    data = json.dumps(config_data)
    r = requests.post(url, data=data, timeout=10000)
    while True:
        if r.status_code==500:
            r = requests.post(url, data=data, timeout=10000)
        else:
            break
    return r.text

def img_request_example(config_yaml):
    url = 'http://127.0.0.1:5000/img_generate'
    #config_yaml="config_img.yaml')"
    with open(config_yaml, "r") as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    print(config_data)
    data = json.dumps(config_data)
    r = requests.post(url, data=data, timeout=100000)
    return r.text

def pretrain_img_request_example(config_yaml):
    url = 'http://127.0.0.1:5000/pretrain_img_generate'
    #config_yaml="config_img.yaml')"
    with open(config_yaml, "r") as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    print(config_data)
    data = json.dumps(config_data)
    r = requests.post(url, data=data, timeout=100000)
    return r.text
    
#pretrain_img_request_example("/home/featurize/work/LLMEthicsPatches/config_img.yaml")

edit_request_example('/home/featurize/work/LLMEthicsPatches/atk_configs/atk-1-mickey.yaml')
edit_request_example('/home/featurize/work/LLMEthicsPatches/atk_configs/atk-2-fight.yaml')
edit_request_example('/home/featurize/work/LLMEthicsPatches/atk_configs/atk-3-bloody.yaml')
edit_request_example('/home/featurize/work/LLMEthicsPatches/atk_configs/atk-4-horrorSnake.yaml')