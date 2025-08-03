import os

class ModeratorConfig:
    def __init__(
            self, 
            model_name:str="1.5",
            work_dir:str=None
        ):
        self.model_name = model_name
        if work_dir is not None:
            self.work_dir = work_dir
        else:
            self.work_dir = os.environ.get("ModeratorWorkDir")
        if self.model_name == "sd1.5":
            self.sd_path = "{work_id}/stable-diffusion-v1-5".format(
                work_id=self.work_dir
            )
            self.sd_unet_path = "{sd_path}/unet/diffusion_pytorch_model.bin".format(
                sd_path=self.sd_path
            )
            self.pretrain_unet_path = "{sd_path}/unet_backup/unet_original_diffusion_pytorch_model.bin".format(
                sd_path=self.sd_path
            )
        elif self.model_name == "sdxl":
            self.sdxl_path = "{work_id}/stable-diffusion-xl-base-1.0".format(
                work_id=self.work_dir
            )
            self.sd_path = "{sdxl_path}".format(
                sdxl_path=self.sdxl_path
            )
            self.sd_unet_path = "{sd_path}/unet/diffusion_pytorch_model.safetensors".format(
                sd_path=self.sd_path
            )
            self.pretrain_unet_path = "{sd_path}/unet_backup/diffusion_pytorch_model.safetensors".format(
                sd_path=self.sd_path
            )    
        self.data_dir = "{work_id}/Moderator/data".format(
            work_id=self.work_dir
        )
        self.img_dir = "{work_id}/Moderator/data/images".format(
            work_id=self.work_dir
        )
        self.finetuned_models_dir = "{work_id}/Moderator/files/models_finetune".format(
            work_id=self.work_dir
        )
        self.edited_models_dir = "{work_id}/Moderator/files/models_edited".format(
            work_id=self.work_dir
        )
        self.task_vectors_dir = "{work_id}/Moderator/files/task_vectors".format(
            work_id=self.work_dir
        )
