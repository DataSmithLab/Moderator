import torch
from diffusers import UNet2DConditionModel
from lib.task_vector_state_dict import TaskVector
import argparse

parser = argparse.ArgumentParser() 
parser.add_argument("--finetuned_ckpt", type=str, default=None, help="the finetuned state dict")
parser.add_argument("--pretrained_ckpt", type=str, default=None, help="the pretrained state dict")
parser.add_argument("--saved_ckpt", type=str, default=None, help="the task vector edited state dict")

def edit_model(finetuned_checkpoint, pretrained_checkpoint, saved_model_name):
    device = torch.device('cpu')
    # Create the task vector
    task_vector = TaskVector(pretrained_checkpoint, finetuned_checkpoint, device=device)
    print('task vector created success')
    # Negate the task vector
    neg_task_vector = -task_vector
    print('task vector neg success')
    # Apply the task vector
    edited_unet_state_dict = neg_task_vector.apply_to(pretrained_checkpoint, scaling_coef=100)
    torch.save(edited_unet_state_dict, saved_model_name)
    print('edited model saved')

args = parser.parse_args()
finetuned_checkpoint = args.finetuned_ckpt
pretrained_ckpt = args.pretrained_ckpt
saved_ckpt = args.saved_ckpt
edit_model(finetuned_checkpoint, pretrained_ckpt, saved_ckpt)