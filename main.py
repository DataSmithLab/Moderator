from lib.utils_config import exp_config_gen
import requests
import json
import yaml
import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument("square", help="display a square of a given number")
#args = parser.parse_args()

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

def main():
    with open("task.yaml", "r") as yaml_file:
        task_data = yaml.safe_load(yaml_file)
    print(task_data)
    edit_config_filename, img_config_filename, generate_img_folder = exp_config_gen(
        src_content=task_data["src_content"],
        dst_content=task_data["dst_content"],
        src_name=task_data["src_name"],
        dst_name=task_data["dst_name"],
        task_name=str(task_data["task_name"]),
        plot_img_contents=task_data["plot_img_content"],
        method=task_data["method"]
    )
    print("image will be generated to ", generate_img_folder)
    edit_request_example(edit_config_filename)
    img_request_example(img_config_filename)
    
main()