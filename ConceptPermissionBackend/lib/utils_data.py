from lib.utils import fresh_sd
import PIL.Image as Image
import os
from diffusers import StableDiffusionPipeline
import datetime
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
    
def dataset_make(datafolder, filenames, dataset_caption):
    output_file = open(datafolder+'/metadata.jsonl', 'w+')  # 打开文件（如果不存在则创建），使用写入模式
    for image in filenames:
        record_str = '{"file_name": "'+image+'", "text": "'+dataset_caption+'"}'
        print(record_str, file=output_file)  # 将输出写入文件
    output_file.close()  # 关闭文件
    
def generate_input_imgs_multi_prompts(task_vector:dict, sd_unet_path:str, pretrain_unet_path:str, model_id:str):
    fresh_sd(sd_unet_path, pretrain_unet_path)
    make_folder(task_vector['input_data_dir'])
    if task_vector['trained']==1:
        pass
    else:
        '''
        image generate
        '''
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            safety_checker=None,
        )
        pipe = pipe.to("cuda")
        prompts = task_vector['prompts']
        real_prompts = task_vector['real_prompts']
        folder_names=task_vector['names']
        for prompt, folder_name, real_prompt in zip(prompts, folder_names, real_prompts):
            img_filenames = []
            for i in range(task_vector['input_num']//task_vector['gen_img_num_per_prompt']):
                images = pipe(real_prompt, num_images_per_prompt=task_vector['gen_img_num_per_prompt']).images
                for idx, image in enumerate(images): 
                    img_filename = "prompt-"+folder_name+"-time-"+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+"-"+str(idx)+".png"
                    img_filenames.append(img_filename)
                    image.save(task_vector['input_data_dir']+"/"+img_filename)
            dataset_make(task_vector['input_data_dir'], img_filenames, prompt)

def generate_demo_imgs(model_id, gen_num, gen_prompt_list, gen_filename_list, folder_name, num_images_per_prompt:int=1):
    make_folder(folder_name)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        safety_checker=None,
    )
    pipe = pipe.to("cuda")
    image_names = []
    for gen_prompt, gen_filename in zip(gen_prompt_list, gen_filename_list):
        for i in range(gen_num//num_images_per_prompt):
            images = pipe(gen_prompt, num_images_per_prompt=num_images_per_prompt).images
            for idx, image in enumerate(images): 
                image_name = "prompt-"+gen_filename+"-time-"+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+"-"+str(idx)+".png"
                image_names.append(image_name)
                image.save(folder_name+"/"+image_name)
    return image_names

def generate_imgs(model_id, sd_unet_path:str, pretrain_unet_path:str, prompt:str, data_folder:str, img_num:int, num_images_per_prompt:int=1):
    #img_num=20
    fresh_sd(sd_unet_path, pretrain_unet_path)
    make_folder(data_folder)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        safety_checker=None,
    )
    pipe = pipe.to("cuda")
    for i in range(img_num//num_images_per_prompt):
        images = pipe(prompt, num_images_per_prompt=num_images_per_prompt).images
        for idx, image in enumerate(images): 
            image.save(data_folder+"/prompt-time-"+prompt+"-"+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+"-"+str(idx)+".png")
    dataset_make(task_vector['input_data_dir'], os.listdir(task_vector['input_data_dir']), prompt)
    
def generate_input_imgs(task_vector:dict, sd_unet_path:str, pretrain_unet_path:str, model_id:str, modified_prompt:str=None):
    fresh_sd(sd_unet_path, pretrain_unet_path)
    make_folder(task_vector['input_data_dir'])
    if task_vector['trained']==1:
        pass
    else:
        '''
        image generate
        '''
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            safety_checker=None,
        )
        pipe = pipe.to("cuda")
        for i in range(task_vector['input_num']//task_vector['gen_img_num_per_prompt']):
            prompt = task_vector['prompt']
            folder_name=task_vector['name']
            images = pipe(prompt, num_images_per_prompt=task_vector['gen_img_num_per_prompt']).images
            for idx, image in enumerate(images): 
                image.save(task_vector['input_data_dir']+"/prompt-"+folder_name+"-time-"+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+"-"+str(idx)+".png")
        if modified_prompt!=None:
            dataset_make(task_vector['input_data_dir'], os.listdir(task_vector['input_data_dir']), modified_prompt)
        else:
            dataset_make(task_vector['input_data_dir'], os.listdir(task_vector['input_data_dir']), prompt)