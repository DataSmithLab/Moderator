import torch
from lib.task_vector import TaskVector
import os
from lib.utils_merge import merge_main
import safetensors

from safetensors.torch import load_file, save_file
from library.model_util import load_models_from_stable_diffusion_checkpoint, load_checkpoint_with_text_encoder_conversion, create_unet_diffusers_config, convert_ldm_unet_checkpoint

def finetuned_unet_extract_safetensors(output_model_name, finetuned_unet_name, backup_unet_name):
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

def get_task_vector(finetuned_checkpoint, pretrained_checkpoint, operator, safetensors=False):
    device = torch.device('cpu')
    # Create the task vector
    task_vector = TaskVector(pretrained_checkpoint, finetuned_checkpoint, device=device, safetensors=safetensors)
    # Negate the task vector
    if operator=="-":
        return -task_vector
    else:
        return task_vector
    
def merge_task_vectors(task_vectors):
    redundant = 'topk20'
    elect = 'mass'
    agg = 'dis-mean'
    scale = 'linear+0.8+2.51+0.1'
    merge_func = redundant+"_"+elect+"_"+agg+"_"+scale
    tv_filenames = []
    for task_vector in task_vectors:
        task_vector_path=task_vector['full_task_vector_path']
        tv_filenames.append(task_vector_path)
    merged_task_vector = merge_main(merge_func, tv_filenames)
    return merged_task_vector

def save_task_vector(pretrained_ckpt, task_vector, model_name="1.5"):
    if task_vector['saved'] & os.path.exists(task_vector['full_task_vector_path']):
        print(task_vector['full_task_vector_path'], "already")
    else:
        if model_name=="1.5":
            finetuned_ckpt=task_vector['finetuned_unet_path']
            operator=task_vector['operator']
            scale=task_vector['scale']
            task_vector_path=task_vector['full_task_vector_path']
            tmp_task_ckpt_vector = get_task_vector(finetuned_ckpt, pretrained_ckpt, operator)
            tmp_task_ckpt_vector = tmp_task_ckpt_vector*scale
            tmp_task_ckpt_vector.vector_save(task_vector_path)
        elif model_name == "xl":
            output_model_ckpt = task_vector['finetuned_model_dir']+"/output_model.safetensors"
            finetuned_ckpt = task_vector['finetuned_model_dir']+"/finetuned_unet.safetensors"
            backup_ckpt = "/root/autodl-fs/stable-diffusion-xl-base-1.0/unet_backup/diffusion_pytorch_model.safetensors"
            
            if os.path.exists(finetuned_ckpt):
                print("Unet extraction already")
            else:
                finetuned_unet_extract_safetensors(output_model_ckpt, finetuned_ckpt, backup_ckpt)
            
            operator=task_vector['operator']
            scale=task_vector['scale']
            task_vector_path=task_vector['full_task_vector_path']
            tmp_task_ckpt_vector = get_task_vector(finetuned_ckpt, pretrained_ckpt, operator, safetensors=True)
            tmp_task_ckpt_vector = tmp_task_ckpt_vector*scale
            tmp_task_ckpt_vector.vector_save(task_vector_path)

def save_task_vectors(pretrained_ckpt, task_vectors, model_name="1.5"):
    for task_vector in task_vectors:
        save_task_vector(pretrained_ckpt, task_vector, model_name=model_name)
        
    
def accumlate_task_vectors(task_vectors):
    init_flag=0
    for task_vector in task_vectors:
        task_vector_path=task_vector['full_task_vector_path']
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
    
def task_vector_apply(pretrained_ckpt, saved_model, task_vectors, merge:bool=False, model_name="1.5"):
    save_task_vectors(pretrained_ckpt, task_vectors, model_name)
    print("all vectors extraced and saved")
    
    if merge:
        final_task_vector=merge_task_vectors(task_vectors)
    else:
        final_task_vector=accumlate_task_vectors(task_vectors)
        
    if model_name == "1.5":
        edited_unet_state_dict = final_task_vector.apply_to(pretrained_ckpt, scaling_coef=1.0, safetensors=False)
    else:
        edited_unet_state_dict = final_task_vector.apply_to(pretrained_ckpt, scaling_coef=1.0, safetensors=True)
    
    if model_name=="1.5":
        torch.save(edited_unet_state_dict, saved_model)
    elif model_name=="xl":
        safetensors.torch.save_file(edited_unet_state_dict, saved_model)
    print("final task vector saved")