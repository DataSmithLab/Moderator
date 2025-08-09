import torch
import json
import numpy as np
import pickle
from safetensors.torch import load_file, save_file

class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None, device=None, vector_path=None, safetensors=False):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        self.device = device
        self.vector = {}
        if vector is not None:
            self.vector = vector
        elif vector_path is not None:
            self.vector = self.vector_load(vector_path)
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                if safetensors:
                    pretrained_state_dict = load_file(pretrained_checkpoint)
                    finetuned_state_dict = load_file(finetuned_checkpoint)
                else:
                    pretrained_state_dict = torch.load(pretrained_checkpoint, map_location=device)
                    #print('pretrain loaded')
                    finetuned_state_dict = torch.load(finetuned_checkpoint, map_location=device)
                    #print('finetune loaded')
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    
    def vector_save(self, vector_path):
        np.save(vector_path, self.vector)
    
    def vector_save_pkl(self, vector_path):
        f_save = open(vector_path, 'wb')
        pickle.dump(self.vector, f_save)
        f_save.close()
    
    def vector_load(self, vector_path):
        load_dict=np.load(vector_path, allow_pickle=True).item()
        return load_dict
        
    
    def __mul__(self, scale):
        new_vector = {}
        with torch.no_grad():
            for key in self.vector:
                new_vector[key] = scale * self.vector[key]
        return TaskVector(vector=new_vector)
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0, safetensors=False):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            if safetensors:
                pretrained_state_dict = load_file(pretrained_checkpoint)
            else:
                pretrained_state_dict = torch.load(pretrained_checkpoint, map_location=self.device)
            new_state_dict = {}
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        return new_state_dict

    def vector_apply(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_state_dict = torch.load(pretrained_checkpoint, map_location=self.device)
            new_state_dict = {}
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        return new_state_dict