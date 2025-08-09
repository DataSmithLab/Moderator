from moderator.src.model_manager import ModelManager
from moderator.src.dataset_manager import DatasetManager
from moderator.src.finetune_manager import FinetuneManager
from moderator.src.task_vector_manager import TaskVectorManager
from moderator.src.configs.moderator_config import ModeratorConfig
from moderator.src.configs.task_vector_config import TVConfig
from moderator.src.configs.experiment_config import ExperimentConfig
from moderator.src.configs.image_config import ImageConfig
import os

class ModeratorManager:
    def __init__(
            self,
            moderator_config:ModeratorConfig
    ) -> None:
        self.moderator_config = moderator_config
        self.model_manager = ModelManager(moderator_config)
        self.dataset_manager = DatasetManager(moderator_config)
        self.finetune_manager = FinetuneManager(moderator_config)
        self.task_vector_manager = TaskVectorManager(moderator_config)
    
    def get_task_vector(self, task_vector_config:TVConfig):
        self.finetune_manager.finetune_on_task(
            task_vector_config
        )
        self.task_vector_manager.save_task_vector(
            task_vector_config
        )

    def edit_model(self, exp_config:ExperimentConfig):
        task_vectors_configs = exp_config.task_vectors_configs

        for task_vector_config in task_vectors_configs:
            self.get_task_vector(task_vector_config)
        self.task_vector_manager.task_vector_apply(
            exp_config.edited_unet_path, 
            task_vectors_configs, 
            merge=exp_config.merge
        )
        return exp_config.edited_unet_path

    def generate_pretrain_image(self, image_config:ImageConfig):
        self.model_manager.generate_images_pretrain(
            image_config
        )
        return image_config.folder_path

    def generate_image(self, image_config:ImageConfig):
        self.model_manager.generate_images(
            image_config
        )
        return image_config.folder_path
