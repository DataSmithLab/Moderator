from typing import List
from moderator.src.context_desc import ContextDesc
from moderator.src.moderator_config import ModeratorConfig
from moderator.src.task_vector_config import TVConfigGenerator

class ExperimentConfig:
    def __init__(
            self,
            task_name:str,
            src_content:ContextDesc,
            dst_content:ContextDesc,
            method:str,
            moderator_config:ModeratorConfig
        ) -> None:
        self.task_name = task_name
        self.src_content = src_content
        self.dst_content = dst_content
        self.method = method
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

    def generate_task_vectors(self):
        if self.method=="replace":
            task_vectors_configs = self.tv_config_generator.replace_tvs(
                self.src_content, 
                self.dst_content, 
                self.src_tv_name, 
                self.dst_tv_name
            )
        elif self.method == "block":
            task_vectors_configs = self.tv_config_generator.neg_tvs(
                self.src_content, 
                None, 
                self.src_tv_name, 
                None
            )
        return task_vectors_configs