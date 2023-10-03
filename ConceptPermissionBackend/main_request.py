import requests
import json
import yaml


def edit_request_example():
    url = 'http://127.0.0.1:5000/model_edit'
    config_yaml="config_edit.yaml"
    with open(config_yaml, "r") as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    data = json.dumps(config_data)
    r = requests.post(url, data=data, timeout=10000)
    return r.text

def img_request_example():
    url = 'http://127.0.0.1:5000/img_generate'
    config_yaml="config_img.yaml"
    with open(config_yaml, "r") as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    data = json.dumps(config_data)
    r = requests.post(url, data=data, timeout=10000)
    return r.text

response_txt = edit_request_example()
#response_txt = img_request_example()
print(response_txt)