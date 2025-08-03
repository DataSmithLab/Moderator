from moderator.src.task_vector import TaskVector
from moderator.src.configs.task_vector_config import TVConfig
from moderator.src.configs.moderator_config import ModeratorConfig
from moderator.src.dataset_manager import DatasetManager
from moderator.src.model_manager import ModelManager
import toml
import subprocess


class FinetuneManager:
    def __init__(
            self, 
            moderator_config:ModeratorConfig
        ) -> None:
        self.moderator_config = moderator_config
        self.dataset_manager = DatasetManager(
            moderator_config=self.moderator_config
        )
        self.model_manager = ModelManager(
            moderator_config=self.moderator_config
        )

    def finetune_model_xl(self, task_vector_config:TVConfig):
        self.model_manager.make_folder(task_vector_config.finetuned_model_folder_path)
        toml_path=task_vector_config.finetuned_model_toml_path

        script_path = self.moderator_config.work_dir+"/Moderator/sd-scripts/sdxl_train.py"
        args = [
            "--pretrained_model_name_or_path", self.moderator_config.sd_path,
            "--output_dir", task_vector_config.finetuned_model_folder_path,
            "--output_name", "output_model",
            "--dataset_config", toml_path,
            "--save_model_as", "safetensors",
            "--learning_rate", "1e-6",
            "--max_train_steps", task_vector_config.train_step,
            "--use_8bit_adam", "--xformers",
            "--gradient_checkpointing",
            "--mixed_precision", "fp16",
            "--cache_latents", "--no_half_vae"
        ]
        command = ["python", script_path] + args
        command_str = " ".join(args)
        print(command_str)
        subprocess.run(command, check=True)
        
        #new_task_vector = TaskVector()
        #new_task_vector.
        #finetuned_unet_extract_safetensors(task_vector['finetuned_model_dir']+"/output_model.safetensors", task_vector['finetuned_model_dir']+"/finetuned_unet.safetensors", WORK_DIR+"/stable-diffusion-xl-base-1.0/unet_backup/diffusion_pytorch_model.safetensors")

    def finetune_model(self, task_vector_config:TVConfig, model_name):
        self.model_manager.make_folder(task_vector_config.finetuned_model_folder_path)
        script_path = "lib/train_text_to_image.py"
        args = [
                "--pretrained_model_name_or_path", model_name,
                "--train_data_dir", task_vector_config.image_config.folder_path,
                "--resolution", "512",
                "--center_crop",
                "--random_flip",
                "--train_batch_size", "1",
                "--use_ema",
                "--gradient_accumulation_steps", "1",
                "--gradient_checkpointing",
                "--mixed_precision", "no",
                "--max_train_steps", task_vector_config.train_step,
                "--learning_rate", "1e-05",
                "--max_grad_norm", "1",
                "--lr_scheduler", "constant",
                "--lr_warmup_steps", "0",
                "--output_dir",  task_vector_config.finetuned_model_folder_path,
                "--validation_prompt", task_vector_config.name,
                "--enable_xformers_memory_efficient_attention",
                "--checkpointing_steps", "10000"
        ]
        command = ["python", script_path] + args
        command_str = " ".join(args)
        print(command_str)
        subprocess.run(command, check=True)
        torch.cuda.empty_cache()

    def finetune_on_task(self, task_vector_config:TVConfig):
        self.dataset_manager.generate_tv_dataset(
            task_vector_config=task_vector_config,
            model_manager=self.model_manager
        )
        torch.cuda.empty_cache()
            
        if self.moderator_config.model_name=="1.5":
            self.finetune_model(
                task_vector_config, 
                self.moderator_config.sd_path
            )
        elif self.moderator_config.model_name == "xl":
            self.finetune_model_xl(
                task_vector_config, 
                self.moderator_config.sd_path
            )