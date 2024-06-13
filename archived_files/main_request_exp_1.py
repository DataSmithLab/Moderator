import requests
import json
import yaml


def edit_request_example(config_yaml):
    url = 'http://127.0.0.1:6000/model_edit'
    #config_yaml="config_edit.yaml"
    with open(config_yaml, "r") as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    data = json.dumps(config_data)
    r = requests.post(url, data=data, timeout=10000)
    return r.text

def img_request_example(config_yaml):
    url = 'http://127.0.0.1:6000/img_generate'
    #config_yaml="config_img.yaml"
    with open(config_yaml, "r") as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    data = json.dumps(config_data)
    r = requests.post(url, data=data, timeout=100000)
    return r.text

def exp_1():
    for prompt in ["gun", "weapon"]:
        for img_num in [100, 200, 1000]:
            for train_step in [1000, 2000]:
                for scale in [1.0, 2.0, 5.0]:
                    config_yaml = "/root/autodl-fs/LLMEthicsPatches/configs/exp-1/"+prompt+"/config_edit_"+prompt+"-img_num_"+str(img_num)+"-scale_"+str(scale)+"-train_step_"+str(train_step)+".yaml"
                    response_txt = edit_request_example(config_yaml)
                    print(prompt, img_num, train_step, scale, " exp-1 edit complete", response_txt)

                    config_yaml = "/root/autodl-fs/LLMEthicsPatches/configs/exp-1/"+prompt+"/config_img_"+prompt+"-img_num_"+str(img_num)+"-scale_"+str(scale)+"-train_step_"+str(train_step)+".yaml"
                    response_txt = img_request_example(config_yaml)
                    print(prompt, img_num, train_step, scale, " exp-1 img complete", response_txt)

def exp_4():
    # exp-4: compare merge with add
    # add with each vector scale=0.1
    response_txt = edit_request_example("/root/autodl-fs/LLMEthicsPatches/configs/exp-4/config_edit_gambling_add_0.1.yaml")
    response_txt = img_request_example("/root/autodl-fs/LLMEthicsPatches/configs/exp-4/config_img_gambling_add_0.1.yaml")
    # add with each vector scale=1.0
    response_txt = edit_request_example("/root/autodl-fs/LLMEthicsPatches/configs/exp-4/config_edit_gambling_add_1.0.yaml")
    response_txt = img_request_example("/root/autodl-fs/LLMEthicsPatches/configs/exp-4/config_img_gambling_add_1.0.yaml")
    # merge
    response_txt = edit_request_example("/root/autodl-fs/LLMEthicsPatches/configs/exp-4/config_edit_gambling_add_merge.yaml")
    response_txt = img_request_example("/root/autodl-fs/LLMEthicsPatches/configs/exp-4/config_img_gambling_merge.yaml")
    print("exp-4 complete")


def exp_5():
    # exp-5: different methods:
    # 1. pretrain; 2. block; 3. mosaic; 4. swap with benign concepts
    # 4: concepts: gun-> doll
    response_txt = edit_request_example("/root/autodl-fs/LLMEthicsPatches/configs/exp-5/gun/config_edit_gun2toy.yaml")
    response_txt = img_request_example("/root/autodl-fs/LLMEthicsPatches/configs/exp-5/gun/config_img_gun2toy.yaml")
    print("exp-5 complete")
    
exp_1()