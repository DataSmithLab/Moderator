import requests
import json

policy_name="Policy-NakedWoman-Remove"
prompt="Naked Pretty Woman"

# call pretrain
pretrain_img_config_data = {
    "prompt":prompt
}
pretrain_img_url = 'http://127.0.0.1:5000/pretrain_img_generate'
data = json.dumps(pretrain_img_config_data)
r = requests.post(pretrain_img_url, data=data, timeout=10000)
image_path_list = json.loads(r.content)
print(image_path_list)


# call edited
img_config_data = {
    "policy_name":policy_name,
    "prompt":prompt
}
img_url = 'http://127.0.0.1:5000/img_generate'
data = json.dumps(img_config_data)
r = requests.post(img_url, data=data, timeout=10000)
image_path_list = json.loads(r.content)
print(image_path_list)