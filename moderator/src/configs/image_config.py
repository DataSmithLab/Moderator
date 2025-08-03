from typing import List
from moderator.src.configs.moderator_config import ModeratorConfig


class ImageConfig:
    def __init__(
            self, 
            moderator_config:ModeratorConfig,
            folder_name:str, 
            img_gen_num:int, 
            img_names:List[str], 
            img_real_prompts:List[str], 
            img_label_prompts:List[str],
            gen_img_num_per_prompt:int=1
        ) -> None:
        self.moderator_config = moderator_config
        self.model_name = moderator_config.model_name

        self.folder_name = folder_name
        self.folder_path = "{img_dir}/{folder_name}".format(
            img_dir = self.moderator_config.img_dir,
            folder_name = self.folder_name
        )

        self.img_gen_num = img_gen_num
        self.img_names = img_names
        self.img_real_prompts = img_real_prompts
        self.img_label_prompts = img_label_prompts
        self.gen_img_num_per_prompt = gen_img_num_per_prompt

    def to_dict(self):
        return {
            "folder_name": self.folder_name,
            "img_gen_num": self.img_gen_num,
            "img_names": self.img_names,
            "img_prompts": self.img_prompts,
            "gen_img_num_per_prompt": self.gen_img_num_per_prompt
        }