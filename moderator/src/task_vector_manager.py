import torch
from moderator.src.task_vector import TaskVector
from moderator.src.configs.moderator_config import ModeratorConfig
from moderator.src.configs.task_vector_config import TVConfig
import os
from typing import List
import safetensors
from safetensors.torch import load_file, save_file
from sdxl_library.model_util import load_checkpoint_with_text_encoder_conversion, create_unet_diffusers_config, convert_ldm_unet_checkpoint


class TaskVectorManager:
    def __init__(self, moderator_config: ModeratorConfig) -> None:
        self.moderator_config = moderator_config

    def finetuned_unet_extract_safetensors(self, output_model_name, finetuned_unet_name, backup_unet_name):
        device="cpu"
        _, state_dict = load_checkpoint_with_text_encoder_conversion(output_model_name, device=device)
        unet_use_linear_projection_in_v2=True
        v2=True
        unet_config = create_unet_diffusers_config(v2, unet_use_linear_projection_in_v2)
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(v2, state_dict, unet_config)
        backup_unet_state_dict = load_file(backup_unet_name)
        converted_unet_checkpoint['add_embedding.linear_1.bias'] = backup_unet_state_dict['add_embedding.linear_1.bias']
        converted_unet_checkpoint['add_embedding.linear_2.bias'] = backup_unet_state_dict['add_embedding.linear_2.bias']
        converted_unet_checkpoint['add_embedding.linear_2.weight'] = backup_unet_state_dict['add_embedding.linear_2.weight']
        converted_unet_checkpoint['add_embedding.linear_1.weight'] = backup_unet_state_dict['add_embedding.linear_1.weight']
        save_file(converted_unet_checkpoint, finetuned_unet_name)

    def get_task_vector(self, finetuned_checkpoint, pretrained_checkpoint, operator, safetensors=False):
        device = torch.device('cpu')
        # Create the task vector
        task_vector = TaskVector(pretrained_checkpoint, finetuned_checkpoint, device=device, safetensors=safetensors)
        # Negate the task vector
        if operator=="-":
            return -task_vector
        else:
            return task_vector

    def save_task_vector(self, task_vector_config: TVConfig):
        if self.moderator_config.model_name == "xl":
            output_model_ckpt = task_vector_config.finetuned_model_folder_path+"/output_model.safetensors"
            finetuned_ckpt = task_vector_config.finetuned_model_folder_path+"/finetuned_unet.safetensors"
            backup_ckpt = self.moderator_config.pretrain_unet_path
                
            if os.path.exists(finetuned_ckpt):
                print("Unet extraction already")
            else:
                self.finetuned_unet_extract_safetensors(output_model_ckpt, finetuned_ckpt, backup_ckpt)
                
            operator=task_vector_config.operator
            scale=task_vector_config.scale
            task_vector_path=task_vector_config.tv_path
            tmp_task_ckpt_vector = self.get_task_vector(
                finetuned_ckpt, 
                self.moderator_config.pretrain_unet_path, 
                operator, 
                safetensors=True
            )
            tmp_task_ckpt_vector = tmp_task_ckpt_vector*scale
            tmp_task_ckpt_vector.vector_save(task_vector_path)

    def save_task_vectors(self, task_vectors_configs: list[TVConfig]):
        for task_vector_config in task_vectors_configs:
            self.save_task_vector(task_vector_config)
        
    def accumlate_task_vectors(self, task_vector_configs: List[TVConfig]):
        init_flag=0
        for task_vector_config in task_vector_configs:
            task_vector_path=task_vector_config.tv_path
            print(task_vector_path)
            tmp_task_vector=TaskVector(vector_path=task_vector_path)
            if init_flag==0:
                final_task_vector=tmp_task_vector
                init_flag=1
                del tmp_task_vector
            else:
                final_task_vector+=tmp_task_vector
                del tmp_task_vector
        return final_task_vector
    
    def task_vector_apply(self, saved_model, task_vectors_configs:List[TVConfig], merge:bool=False):
        self.save_task_vectors(task_vectors_configs)
        
        #if merge:
        #    final_task_vector=merge_task_vectors(task_vectors)
        #else:
        final_task_vector=self.accumlate_task_vectors(task_vectors_configs)
            
        if self.moderator_config.model_name == "1.5":
            edited_unet_state_dict = final_task_vector.apply_to(
                self.moderator_config.pretrain_unet_path, 
                scaling_coef=1.0, 
                safetensors=False
            )
        else:
            edited_unet_state_dict = final_task_vector.apply_to(
                self.moderator_config.pretrain_unet_path, 
                scaling_coef=1.0, 
                safetensors=True
            )
        
        if self.moderator_config.model_name=="1.5":
            torch.save(edited_unet_state_dict, saved_model)
        elif self.moderator_config.model_name=="xl":
            safetensors.torch.save_file(edited_unet_state_dict, saved_model)