import torch
from lib.task_vector import TaskVector
import os

def get_task_vector(finetuned_checkpoint, pretrained_checkpoint, operator):
    device = torch.device('cpu')
    # Create the task vector
    task_vector = TaskVector(pretrained_checkpoint, finetuned_checkpoint, device=device)
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

def save_task_vector(pretrained_ckpt, task_vector):
    if task_vector['saved'] & os.path.exists(task_vector['full_task_vector_path']):
        print(task_vector['full_task_vector_path'], "already")
    else:
        finetuned_ckpt=task_vector['finetuned_unet_path']
        operator=task_vector['operator']
        scale=task_vector['scale']
        task_vector_path=task_vector['full_task_vector_path']
        tmp_task_ckpt_vector = get_task_vector(finetuned_ckpt, pretrained_ckpt, operator)
        tmp_task_ckpt_vector = tmp_task_ckpt_vector*scale
        tmp_task_ckpt_vector.vector_save(task_vector_path)

def save_task_vectors(pretrained_ckpt, task_vectors):
    for task_vector in task_vectors:
        save_task_vector(pretrained_ckpt, task_vector)
        
    
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
    
def task_vector_apply(pretrained_ckpt, saved_model, task_vectors, merge:bool=False):
    save_task_vectors(pretrained_ckpt, task_vectors)
    print("all vectors extraced and saved")
    
    if merge:
        final_task_vector=merge_task_vectors(task_vectors)
    else:
        final_task_vector=accumlate_task_vectors(task_vectors)
        
    edited_unet_state_dict = final_task_vector.apply_to(pretrained_ckpt, scaling_coef=1.0)
    torch.save(edited_unet_state_dict, saved_model)
    print("final task vector saved")