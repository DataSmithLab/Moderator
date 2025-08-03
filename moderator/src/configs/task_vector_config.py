from typing import List
from moderator.src.configs.moderator_config import ModeratorConfig
from moderator.src.context_desc import ContextDesc
from moderator.src.configs.content_config import ContentConfig
from moderator.src.query_expander import QueryExpander
from moderator.src.configs.image_config import ImageConfig


DEFAULT_INPUT_NUM = 50
DEFAULT_TRAIN_STEP = 500

class TVConfig:
    def __init__(
            self, 
            moderator_config:ModeratorConfig,
            name:str,
            operator:str,
            content_config: ContentConfig,
            input_num:int=DEFAULT_INPUT_NUM,
            scale:float=1.0,
            train_step:int=DEFAULT_TRAIN_STEP
        ):
        self.moderator_config = moderator_config
        self.name = name
        self.operator = operator
        self.input_num = input_num
        self.scale = scale
        self.train_step = train_step
        self.content_config = content_config

        self.tv_path = "{task_vectors_dir}/{name}.safetensors".format(
            task_vectors_dir = self.moderator_config.task_vectors_dir,
            name = self.name
        )

        self.tv_finetune_image_folder_path = "{img_dir}/{name}.png".format(
            img_dir = self.moderator_config.img_dir,
            name = self.name
        )

        self.image_config: ImageConfig = None
    
    def __str__(self) -> str:
        return '''
        TVConfig(
            name={name}, 
            operator={operator}, 
            input_num={input_num}, 
            scale={scale}, 
            train_step={train_step}, 
            content_config={content_config}
        )'''.format(
            name = self.name,
            operator = self.operator,
            input_num = self.input_num,
            scale = self.scale,
            train_step = self.train_step,
            content_config = self.content_config.__str__()
        )
    
    def set_image_config(self, image_config):
        self.image_config = image_config   

    def to_dict(self):
        return {
            "name": self.name,
            "operator": self.operator,
            "input_num": self.input_num,
            "scale": self.scale,
            "train_step": self.train_step,
            "content_config": self.content_config.to_dict()
        }

class TVConfigGenerator:
    def __init__(self, moderator_config: ModeratorConfig) -> None:
        self.moderator_config = moderator_config
        self.query_expander = QueryExpander()

    def add_tv(
            self,
            image_content, 
            label_content, 
            tv_name
        ):
        content_config = ContentConfig(
            content_name = tv_name,
            label_content = label_content,
            real_content = image_content
        )
        tv_config = TVConfig(
            moderator_config = self.moderator_config,
            name = tv_name,
            operator = '+',
            content_config = content_config
        )
        return tv_config

    def neg_tv(
            self,
            image_content, 
            label_content, 
            tv_name
        ):
        content_config = ContentConfig(
            content_name = tv_name,
            label_content = label_content,
            real_content = image_content
        )
        tv_config = TVConfig(
            moderator_config = self.moderator_config,
            name = tv_name,
            operator = '-',
            content_config = content_config
        )
        return tv_config
    
    def generate_image_config(
            self,
            tv_name:str,
            input_num:int,
            image_real_prompts:List[str],
            image_label_prompts:List[str]
        )->ImageConfig:
        image_names = [
            "{tv_name}_{i}".format(
                tv_name = tv_name,
                i = i
            ) for i in range(input_num)
        ]
        image_config = ImageConfig(
            moderator_config = self.moderator_config,
            folder_name = tv_name,
            img_gen_num = input_num,
            img_names = image_names,
            img_real_prompts = image_real_prompts,
            img_label_prompts = image_label_prompts,
            gen_img_num_per_prompt = 1
        )
        return image_config

    def replace_tvs(
            self,
            src_content, 
            dst_content, 
            src_tv_name, 
            dst_tv_name,
            expand_key,
            expand_type
        ):
        real_prompt_list, swap_prompt_list = self.query_expander.overall_expansion(
            input_context_desc=src_content,
            swap_context_desc=dst_content,
            expand_key=expand_key,
            expand_type=expand_type
        )

        neg_src_tv = self.neg_tv(
            image_content = src_content,
            label_content = src_content,
            tv_name = src_tv_name
        )
        neg_src_tv_image_config = self.generate_image_config(
            tv_name = src_tv_name,
            input_num = neg_src_tv.input_num,
            image_real_prompts = real_prompt_list,
            image_label_prompts = real_prompt_list
        )
        neg_src_tv.set_image_config(
            image_config=neg_src_tv_image_config
        )

        add_dst_tv = self.add_tv(
            image_content = dst_content,
            label_content = src_content,
            tv_name = dst_tv_name
        )
        add_dst_tv_image_config = self.generate_image_config(
            tv_name = dst_tv_name,
            input_num = add_dst_tv.input_num,
            image_real_prompts = swap_prompt_list,
            image_label_prompts = real_prompt_list
        )
        add_dst_tv.set_image_config(
            image_config=add_dst_tv_image_config
        )
        return [neg_src_tv, add_dst_tv]

    def neg_tvs(
            self,
            src_content=None, 
            father_content=None, 
            src_tv_name=None, 
            father_tv_name=None,
            expand_key=None,
            expand_type=None
        ):

        real_prompt_list= self.query_expander.overall_expansion(
            input_context_desc=src_content,
            expand_key=expand_key,
            expand_type=expand_type
        )
        neg_src_tv = self.neg_tv(
            image_content = src_content,
            label_content = src_content,
            tv_name = src_tv_name
        )
        neg_src_tv_image_config = self.generate_image_config(
            tv_name = src_tv_name,
            input_num = neg_src_tv.input_num,
            image_real_prompts = real_prompt_list,
            image_label_prompts = real_prompt_list
        )
        neg_src_tv.set_image_config(
            image_config=neg_src_tv_image_config
        )

        if father_content is None or father_content!="nan":
            return [neg_src_tv]
        else:
            father_prompt_list= self.query_expander.overall_expansion(
                input_context_desc=father_content,
                expand_key=expand_key,
                expand_type=expand_type
            )
            add_father_tv = self.add_tv(
                image_content = father_content,
                label_content = father_content,
                tv_name = father_tv_name
            )
            add_father_tv_image_config = self.generate_image_config(
                tv_name = father_tv_name,
                input_num = add_father_tv.input_num,
                image_real_prompts = father_prompt_list,
                image_label_prompts = father_prompt_list
            )
            add_father_tv.set_image_config(
                image_config=add_father_tv_image_config
            )
            return [neg_src_tv, add_father_tv]