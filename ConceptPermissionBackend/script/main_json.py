import yaml
import json
with open('/root/autodl-fs/LLMEthicsPatches/configs/permission_interfaces_setting.yaml', "r") as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)
idx = 1
category_map = {}
for word_type, word_list in yaml_data.items():
    category_map[word_type]=word_list
print(category_map)