from lib.utils import fresh_sd
import PIL.Image as Image
import os
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
import datetime
from lib.utils_query_expansion import QueryExpansion

import shutil

    

#query_expansion = QueryExpansion(model_name="llama3")
query_expansion = QueryExpansion(model_name="vicuna")
# 定义图像拼接函数
def image_compose(images_path, image_names, image_column, image_row, image_size, image_save_path):
    to_image = Image.new('RGB', (image_column * image_size, image_row * image_size))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, image_row + 1):
        for x in range(1, image_column + 1):
            from_image = Image.open(images_path + image_names[image_column * (y - 1) + x - 1]).resize(
                (image_size, image_size), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * image_size, (y - 1) * image_size))
    return to_image.save(image_save_path)  # 保存新图

def make_folder(folder_path):
    if not os.path.exists(folder_path):  # 检查文件夹是否存在
        os.mkdir(folder_path)  # 创建文件夹
        print(f"文件夹已创建: {folder_path}")
    else:
        print(f"文件夹已存在: {folder_path}")
    
def dataset_make(datafolder, filenames, dataset_captions):
    output_file = open(datafolder+'/metadata.jsonl', 'w+')  # 打开文件（如果不存在则创建），使用写入模式
    for image, dataset_caption in zip(filenames, dataset_captions):
        record_str = '{"file_name": "'+image+'", "text": "'+dataset_caption+'"}'
        print(record_str, file=output_file)  # 将输出写入文件
    output_file.close()  # 关闭文件
    
def apply_mosaic(image_path, output_path):
    # 打开图像
    img = Image.open(image_path)
    width, height = img.size
    
    # 计算中心区域的坐标
    center_x, center_y = width // 2, height // 2
    left = center_x - width // 4
    upper = center_y - height // 4
    right = center_x + width // 4
    lower = center_y + height // 4
    
    # 获取中心区域
    center_region = img.crop((left, upper, right, lower))
    
    # 将中心区域打马赛克
    mosaic_size = 10  # 可以调整这个值来改变马赛克的颗粒大小
    small = center_region.resize((center_region.width // mosaic_size, center_region.height // mosaic_size), resample=Image.NEAREST)
    mosaic = small.resize(center_region.size, Image.NEAREST)
    
    # 将马赛克区域粘贴回原图
    img.paste(mosaic, (left, upper))
    
    # 保存处理后的图像
    img.save(output_path)
    
def generate_input_imgs_multi_prompts(task_vector:dict, sd_unet_path:str, pretrain_unet_path:str, model_id:str, model_name="1.5"):
    print(pretrain_unet_path)
    fresh_sd(sd_unet_path, pretrain_unet_path)
    make_folder(task_vector['input_data_dir'])
    if task_vector['trained']==1:
        pass
    else:
        '''
        image generate
        '''
        if model_name == "1.5":
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, 
                safety_checker=None,
            )
        elif model_name == "xl":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                safety_checker=None
            )
        pipe = pipe.to("cuda")
        
        
        #for images_config in task_vector["images_configs"]:
        if True:
            images_config = task_vector["images_configs"]
            label_context = images_config['label_context']
            real_context = images_config['real_context']
            expand_key = images_config['expand_key']
            expand_type = images_config['expand_type']
            folder_name = images_config['image_name']

            real_prompt_list, swap_prompt_list = query_expansion.overall_expansion(
                input_context_desc=real_context,
                swap_context_desc=label_context,
                expand_1_key=expand_key,
                expand_1_type=expand_type
            )

            img_filenames = []
            #for i in range(task_vector['input_num']//task_vector['gen_img_num_per_prompt']):
            for real_prompt, label_prompt in zip(real_prompt_list, swap_prompt_list):
                images = pipe(real_prompt, num_images_per_prompt=task_vector['gen_img_num_per_prompt'], width=1024, height=1024).images
                for idx, image in enumerate(images): 
                    img_filename = "prompt-"+folder_name+"-time-"+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+"-"+str(idx)+".png"
                    img_filenames.append(img_filename)
                    image.save(task_vector['input_data_dir']+"/"+img_filename)
                    if "mosaic" in task_vector and task_vector["mosaic"]==True:
                        apply_mosaic(task_vector['input_data_dir']+"/"+img_filename, task_vector['input_data_dir']+"/"+img_filename)
            dataset_make(task_vector['input_data_dir'], img_filenames, swap_prompt_list)
            
def generate_demo_imgs(model_id, gen_num, gen_prompt_list, gen_filename_list, folder_name, num_images_per_prompt:int=1, width=512, height=512, model_name="1.5"):
    make_folder(folder_name)
    if model_name == "1.5":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            safety_checker=None,
        )
    elif model_name == "xl":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            safety_checker=None
        )
    pipe = pipe.to("cuda")
    image_names = []
    flask_image_names = []
    for gen_prompt, gen_filename in zip(gen_prompt_list, gen_filename_list):
        gen_folder_path = folder_name+"/"+gen_filename
        if os.path.exists(gen_folder_path):
            pass
        else:
            os.makedirs(gen_folder_path)
        for i in range(gen_num//num_images_per_prompt):
            images = pipe(gen_prompt, num_images_per_prompt=num_images_per_prompt, width=width, height=height).images
            for idx, image in enumerate(images): 
                image_name = "prompt-"+gen_filename+"-time-"+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+"-"+str(idx)+".png"
                image_names.append(image_name)
                image.save(gen_folder_path+"/"+image_name)
                shutil.copy(gen_folder_path+"/"+image_name, "static"+"/"+image_name)
                flask_image_names.append("/static"+"/"+image_name)
    return flask_image_names

def SD_pipe(model_id, model_name="xl"):
    if model_name == "1.5":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            safety_checker=None,
        )
    elif model_name == "xl":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            safety_checker=None
        )
    pipe = pipe.to("cuda")
    return pipe