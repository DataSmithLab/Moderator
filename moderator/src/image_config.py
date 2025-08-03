from typing import List
from moderator.src.moderator_config import ModeratorConfig


class ImageConfig:
    def __init__(
            self, 
            moderator_config:ModeratorConfig,
            folder_name:str, 
            img_gen_num:int, 
            img_names:List[str], 
            img_prompts:List[str], 
            gen_img_num_per_prompt:int
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
        self.img_prompts = img_prompts
        self.gen_img_num_per_prompt = gen_img_num_per_prompt