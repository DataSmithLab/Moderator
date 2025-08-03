from typing import List
from moderator.src.moderator_config import ModeratorConfig
from moderator.src.context_desc import ContextDesc
from moderator.src.content_config import ContentConfig

DEFAULT_INPUT_NUM = 50
DEFAULT_TRAIN_STEP = 500

class TVConfig:
    def __init__(
            self, 
            moderator_config:ModeratorConfig,
            name:str,
            operator:str,
            content_config_list: List[ContentConfig],
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
        self.content_config_list = content_config_list

        self.tv_path = "{task_vectors_dir}/{name}.safetensors".format(
            task_vectors_dir = self.moderator_config.task_vectors_dir,
            name = self.name
        )
    
    def __str__(self) -> str:
        return '''
        TVConfig(
            name={name}, 
            operator={operator}, 
            input_num={input_num}, 
            scale={scale}, 
            train_step={train_step}, 
            content_config_list={content_config_list}
        )'''.format(
            name = self.name,
            operator = self.operator,
            input_num = self.input_num,
            scale = self.scale,
            train_step = self.train_step,
            content_config_list = [content_config.__str__() for content_config in self.content_config_list]
        )

    def to_dict(self):
        return {
            "name": self.name,
            "operator": self.operator,
            "input_num": self.input_num,
            "scale": self.scale,
            "train_step": self.train_step,
            "content_config_list": [content_config.to_dict() for content_config in self.content_config_list]
        }

class TVConfigGenerator:
    def __init__(self, moderator_config: ModeratorConfig) -> None:
        self.moderator_config = moderator_config

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
            content_config_list = [content_config]
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
            content_config_list = [content_config]
        )
        return tv_config
    
    def replace_tvs(
            self,
            src_content, 
            dst_content, 
            src_tv_name, 
            dst_tv_name
        ):
        neg_src_tv = self.neg_tv(
            image_content = src_content,
            label_content = src_content,
            tv_name = src_tv_name
        )
        add_dst_tv = self.add_tv(
            image_content = dst_content,
            label_content = src_content,
            tv_name = dst_tv_name
        )
        return [neg_src_tv, add_dst_tv]

    def neg_tvs(
            self,
            src_content=None, 
            father_content=None, 
            src_tv_name=None, 
            father_tv_name=None
        ):
        neg_src_tv = self.neg_tv(
            image_content = src_content,
            label_content = src_content,
            tv_name = src_tv_name
        )
        if father_content is None or father_content!="nan":
            return [neg_src_tv]
        else:
            add_father_tv = self.add_tv(
                image_content = father_content,
                label_content = father_content,
                tv_name = father_tv_name
            )
            return [neg_src_tv, add_father_tv]