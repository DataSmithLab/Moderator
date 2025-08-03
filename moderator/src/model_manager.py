import os
from moderator.src.configs.image_config import ImageConfig
from moderator.src.configs.moderator_config import ModeratorConfig
from moderator.src.configs.task_vector_config import TVConfig

import PIL.Image as Image
import os
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
import datetime

import shutil

class ModelManager:
    def __init__(
            self,
            moderator_config:ModeratorConfig
        ) -> None:
        self.moderator_config = moderator_config
        self.runtime_unet_path = self.moderator_config.sd_unet_path
        self.pretrain_unet_path = self.moderator_config.pretrain_unet_path
        self.model_name = self.moderator_config.model_name
        self.model_folder_path = self.moderator_config.sd_path

    def get_sd_pipe(self):
        if self.model_name == "sd1.5":
            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_folder_path, 
                safety_checker=None,
            )
        elif self.model_name == "sdxl":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_folder_path, 
                torch_dtype=torch.float16, 
                safety_checker=None
            )
        pipe = pipe.to("cuda")
        return pipe

    def fresh_sd(
            self,
            fresh_unet_path
        ):
        try:
            os.remove(self.runtime_unet_path)
        except:
            pass
        os.symlink(
            fresh_unet_path, 
            self.runtime_unet_path
        )

    def make_folder(
            self, 
            folder_path:str
        ):
        if not os.path.exists(folder_path):  # 检查文件夹是否存在
            os.mkdir(folder_path)  # 创建文件夹
            print(f"Dataset Folder Created: {folder_path}")
        else:
            print(f"Dataset Folder Exists: {folder_path}")

    def generate_input_imgs_multi_prompts(
            self,
            task_vector_config:TVConfig, 
            sd_unet_path:str
        ):
        self.fresh_sd(
            sd_unet_path
        )
        self.make_folder(task_vector['input_data_dir'])
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