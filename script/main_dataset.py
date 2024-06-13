import os

with open("/root/autodl-fs/LLMEthicsPatches/data/input-Dice-input_num-100/metadata.jsonl", "w") as f:
    for img_filename in os.listdir("/root/autodl-fs/LLMEthicsPatches/data/input-Dice-input_num-100"):
        if "Poker" in img_filename:
            record_str = '{"file_name": "'+img_filename+'", "text": "'+'Poker'+'"}'
            print(record_str, file=f)  # 将输出写入文件
        elif "Dice" in img_filename:
            record_str = '{"file_name": "'+img_filename+'", "text": "'+'Dice'+'"}'
            print(record_str, file=f)  # 将输出写入文件
            

with open("/root/autodl-fs/LLMEthicsPatches/data/input-Dice+Poker-input_num-100/metadata.jsonl", "w") as f:
    for img_filename in os.listdir("/root/autodl-fs/LLMEthicsPatches/data/input-Dice+Poker-input_num-100"):
        if "Poker" in img_filename:
            record_str = '{"file_name": "'+img_filename+'", "text": "'+'Poker'+'"}'
            print(record_str, file=f)  # 将输出写入文件
        elif "Dice" in img_filename:
            record_str = '{"file_name": "'+img_filename+'", "text": "'+'Dice'+'"}'
            print(record_str, file=f)  # 将输出写入文件
            
with open("/root/autodl-fs/LLMEthicsPatches/data/input-Poker-input_num-100/metadata.jsonl", "w") as f:
    for img_filename in os.listdir("/root/autodl-fs/LLMEthicsPatches/data/input-Poker-input_num-100"):
        if "Poker" in img_filename:
            record_str = '{"file_name": "'+img_filename+'", "text": "'+'Poker'+'"}'
            print(record_str, file=f)  # 将输出写入文件
        elif "Dice" in img_filename:
            record_str = '{"file_name": "'+img_filename+'", "text": "'+'Dice'+'"}'
            print(record_str, file=f)  # 将输出写入文件