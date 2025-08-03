import os
from moderator.src.configs.image_config import ImageConfig
from moderator.src.configs.moderator_config import ModeratorConfig
from moderator.src.configs.task_vector_config import TVConfig

import PIL.Image as Image
import os
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch

from moderator.src.dataset_manager import DatasetManager

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

    def generate_images(
            self,
            sd_unet_path:str,
            image_config:ImageConfig
        ):
        self.make_folder(
            folder_path=image_config.folder_path
        )
        self.fresh_sd(
            sd_unet_path
        )
        pipe = self.get_sd_pipe()
        image_filenames = []
        label_prompts = []
        for real_prompt, label_prompt, image_name in zip(image_config.img_real_prompts, image_config.img_label_prompts, image_config.img_names):
            image = pipe(
                real_prompt, 
                num_images_per_prompt=image_config.gen_img_num_per_prompt, 
                width=1024, 
                height=1024
            ).images[0]
            img_filename = image_name+".png"
            image_filenames.append(img_filename)
            image.save(image_config.folder_path+"/"+img_filename)
            label_prompts.append(label_prompt)
        return image_filenames, label_prompts

    def generate_tv_images(
            self,
            task_vector_config:TVConfig, 
            sd_unet_path:str
        ):
        image_filenames, label_prompts = self.generate_images(
            sd_unet_path=sd_unet_path,
            image_config=task_vector_config.image_config
        )
        DatasetManager.dataset_make(
            dataset_dir=task_vector_config.image_config.folder_path,
            image_filenames=image_filenames,
            label_prompts=label_prompts
        )