from typing import List
from moderator.src.context_desc import ContextDesc, build_context_desc_from_dict
from moderator.src.configs.moderator_config import ModeratorConfig
from moderator.src.configs.task_vector_config import TVConfigGenerator
import json

class PolicyConfig:
    def __init__(
            self,
            task_name:str,
            src_content:ContextDesc,
            dst_content:ContextDesc,
            expand_key:str,
            expand_type:str,
            method:str,
            moderator_config:ModeratorConfig
        ) -> None:
        self.task_name = task_name
        self.src_content = src_content
        self.dst_content = dst_content

        self.method = method
        self.expand_key = expand_key
        self.expand_type = expand_type

        self.moderator_config = moderator_config

        self.model_name = self.moderator_config.model_name

        assert self.model_name in ["sdxl", "sd1.5"]
        assert self.method in ["remove", "replace", "mosaic"]

        self.src_tv_name = "tv-obj:{obj}-sty:{sty}-act:{act}".format(
            obj=self.src_content.obj,
            sty=self.src_content.sty,
            act=self.src_content.act
        )
        self.dst_tv_name = "tv-obj:{obj}-sty:{sty}-act:{act}".format(
            obj=self.dst_content.obj,
            sty=self.dst_content.sty,
            act=self.dst_content.act
        )

        self.edited_unet_path = "{edited_models_dir}/{task_name}.safetensors".format(
            edited_models_dir = self.moderator_config.edited_models_dir,
            task_name = self.task_name
        )

        self.tv_config_generator = TVConfigGenerator(
            moderator_config=self.moderator_config
        )

        self.task_vectors_configs = self.generate_task_vectors()

        self.merge=False #TODO

    def __str__(self) -> str:
        return '''
        ExperimentConfig(
            task_name={task_name},
            src_content={src_content},
            dst_content={dst_content},
            method={method},
            task_vectors_configs={task_vectors_configs},
            edited_unet_path={edited_unet_path}
        )'''.format(
            task_name=self.task_name,
            src_content=self.src_content.__str__(),
            dst_content=self.dst_content.__str__(),
            method=self.method,
            task_vectors_configs=[tv_config.__str__() for tv_config in self.task_vectors_configs],
            edited_unet_path=self.edited_unet_path
        )

    def to_dict(self):
        return {
            "task_name":self.task_name,
            "src_content":self.src_content.to_dict(),
            "dst_content":self.dst_content.to_dict(),
            "method":self.method,
            "expand_key":self.expand_key,
            "expand_type":self.expand_type,
            "moderator_config":self.moderator_config.to_dict(),
            "task_vectors_configs":[tv_config.to_dict() for tv_config in self.task_vectors_configs],
            "edited_unet_path":self.edited_unet_path,
            "merge":self.merge,
            "src_tv_name":self.src_tv_name,
            "dst_tv_name":self.dst_tv_name
        }

    def generate_task_vectors(self):
        if self.method=="replace":
            task_vectors_configs = self.tv_config_generator.replace_tvs(
                self.src_content, 
                self.dst_content, 
                self.src_tv_name, 
                self.dst_tv_name,
                self.expand_key,
                self.expand_type
            )
        elif self.method == "block":
            task_vectors_configs = self.tv_config_generator.neg_tvs(
                self.src_content, 
                None, 
                self.src_tv_name, 
                None,
                self.expand_key,
                self.expand_type
            )
        return task_vectors_configs
    
def build_policy_config_from_dict(
    exp_config_dict:dict,
    moderator_config:ModeratorConfig
):
    task_name = exp_config_dict["task_name"]
    src_content = build_context_desc_from_dict(exp_config_dict["src_content"])
    dst_content = build_context_desc_from_dict(exp_config_dict["dst_content"])
    expand_key = exp_config_dict["expand_key"]
    expand_type = exp_config_dict["expand_type"]
    method = exp_config_dict["method"]
    return PolicyConfig(
        task_name=task_name,
        src_content=src_content,
        dst_content=dst_content,
        expand_key=expand_key,
        expand_type=expand_type,
        method=method,
        moderator_config=moderator_config
    )