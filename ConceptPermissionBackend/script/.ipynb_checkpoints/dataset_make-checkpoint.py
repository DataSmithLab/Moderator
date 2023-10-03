import json
import os
import shutil
import argparse
import pandas as pd

parser = argparse.ArgumentParser() 
parser.add_argument("--dataset_folder", type=str, default='/root/autodl-fs/LLMEthicsPatches/data/naked_woman', help="the dataset folder for ethics")
parser.add_argument("--dataset_caption", type=str, default='naked, woman', help="the caption of dataset")
args = parser.parse_args()

def make_dataset(datafolder, filenames):
    output_file = open(datafolder+'/metadata.jsonl', 'w')  # 打开文件（如果不存在则创建），使用写入模式
    for image in filenames:
        record_str = '{"file_name": "'+image+'", "text": "'+args.dataset_caption+'"}'
        print(record_str, file=output_file)  # 将输出写入文件
    output_file.close()  # 关闭文件
    
    img_path_list = []
    img_caption_list = []
    for image in filenames:
        img_path_list.append(args.dataset_folder+"/"+image)
        img_caption_list.append(args.dataset_caption)
    output_df = pd.DataFrame({'image_column':img_path_list, 'caption_column':img_caption_list})
    output_df.to_csv(args.dataset_folder+"/"+"dataset.csv", index=False)

make_dataset(args.dataset_folder, os.listdir(args.dataset_folder))