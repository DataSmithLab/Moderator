import torch


class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None, device=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        self.device = device
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_state_dict = torch.load(pretrained_checkpoint, map_location=device)
                print('pretrain loaded')
                finetuned_state_dict = torch.load(finetuned_checkpoint, map_location=device)
                print('finetune loaded')
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    print('pretrain:', pretrained_state_dict[key])
                    print('fintuned:', finetuned_state_dict[key])
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
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

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_state_dict = torch.load(pretrained_checkpoint, map_location=self.device)
            new_state_dict = {}
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        return pretrained_state_dict

