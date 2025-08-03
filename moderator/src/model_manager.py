import os
from moderator.src.configs.image_config import ImageConfig
from moderator.src.configs.moderator_config import ModeratorConfig
from moderator.src.configs.task_vector_config import TVConfig

import PIL.Image as Image
import os
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch

from moderator.src.dataset_manager import DatasetManager

class ModelManager:
    def __init__(
            self,
            moderator_config:ModeratorConfig
        ) -> None:
        self.moderator_config = moderator_config
        self.runtime_unet_path = self.moderator_config.sd_unet_path
        self.pretrain_unet_path = self.moderator_config.pretrain_unet_path
        self.model_name = self.moderator_config.model_name
        self.model_folder_path = self.moderator_config.sd_path

    def get_sd_pipe(self):
        if self.model_name == "sd1.5":
            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_folder_path, 
                safety_checker=None,
            )
        elif self.model_name == "sdxl":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_folder_path, 
                torch_dtype=torch.float16, 
                safety_checker=None
            )
        pipe = pipe.to("cuda")
        return pipe

    def fresh_sd(
            self,
            fresh_unet_path
        ):
        try:
            os.remove(self.runtime_unet_path)
        except:
            pass
        os.symlink(
            fresh_unet_path, 
            self.runtime_unet_path
        )

    def generate_images(
            self,
            sd_unet_path:str,
            image_config:ImageConfig
        ):
        self.make_folder(
            folder_path=image_config.folder_path
        )
        self.fresh_sd(
            sd_unet_path
        )
        pipe = self.get_sd_pipe()
        image_filenames = []
        label_prompts = []
        for real_prompt, label_prompt, image_name in zip(image_config.img_real_prompts, image_config.img_label_prompts, image_config.img_names):
            image = pipe(
                real_prompt, 
                num_images_per_prompt=image_config.gen_img_num_per_prompt, 
                width=1024, 
                height=1024
            ).images[0]
            img_filename = image_name+".png"
            image_filenames.append(img_filename)
            image.save(image_config.folder_path+"/"+img_filename)
            label_prompts.append(label_prompt)
        return image_filenames, label_prompts
    

def xl_finetune_toml_make(task_vector:dict):
    toml_path = task_vector['finetuned_model_dir']+"/fine_tune.toml"
    
    dataset_adjust(task_vector['input_data_dir'])
    
    toml_config = {
        'general': {
            'shuffle_caption': True, 
            'keep_tokens': 1
        },
        'datasets': [
            {
                'resolution': [512, 512],
                'batch_size': 1,
                'subsets': [
                    {
                        'image_dir': task_vector['input_data_dir'],
                        'metadata_file': task_vector['input_data_dir']+'/metadata.json'
                    }
                ]
            }
        ]
    }
    with open(toml_path, 'w+') as f:
        toml.dump(toml_config, f)
    return toml_path

def finetuned_unet_extract_xl(finetuned_model_name, finetuned_unet_name, backup_unet_name):
    device="cpu"
    _, state_dict = load_checkpoint_with_text_encoder_conversion(finetuned_model_name, device=device)
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

def finetune_model_xl(task_vector:dict, model_id):
    make_folder(task_vector['finetuned_model_dir'])
    toml_path=xl_finetune_toml_make(task_vector)
    script_path = WORK_DIR+"/ConceptPermission/sd-scripts/sdxl_train.py"
    args = [
        "--pretrained_model_name_or_path", model_id,
        "--output_dir", task_vector['finetuned_model_dir'],
        "--output_name", "output_model",
        "--dataset_config", toml_path,
        "--save_model_as", "safetensors",
        "--learning_rate", "1e-6",
        "--max_train_steps", str(task_vector['train_step']),
        "--use_8bit_adam", "--xformers",
        "--gradient_checkpointing",
        "--mixed_precision", "fp16",
        "--cache_latents", "--no_half_vae"
    ]
    command = ["python", script_path] + args
    command_str = " ".join(args)
    print(command_str)
    subprocess.run(command, check=True)
    finetuned_unet_extract_safetensors(task_vector['finetuned_model_dir']+"/output_model.safetensors", task_vector['finetuned_model_dir']+"/finetuned_unet.safetensors", WORK_DIR+"/stable-diffusion-xl-base-1.0/unet_backup/diffusion_pytorch_model.safetensors")

def finetune_model(task_vector:dict, model_name):
    make_folder(task_vector['finetuned_model_dir'])
    script_path = "lib/train_text_to_image.py"
    args = [
            "--pretrained_model_name_or_path", model_name,
            "--train_data_dir", task_vector['input_data_dir'],
            "--resolution", "512",
            "--center_crop",
            "--random_flip",
            "--train_batch_size", "1",
            "--use_ema",
            "--gradient_accumulation_steps", "1",
            "--gradient_checkpointing",
            "--mixed_precision", "no",
            "--max_train_steps", str(task_vector['train_step']),
            "--learning_rate", "1e-05",
            "--max_grad_norm", "1",
            "--lr_scheduler", "constant",
            "--lr_warmup_steps", "0",
            "--output_dir",  task_vector['finetuned_model_dir'],
            "--validation_prompt", task_vector['name'],
            "--enable_xformers_memory_efficient_attention",
            "--checkpointing_steps", "10000"
    ]
    command = ["python", script_path] + args
    command_str = " ".join(args)
    print(command_str)
    subprocess.run(command, check=True)
    torch.cuda.empty_cache()
    print('after finetune_model', torch.cuda.memory_summary())

def finetune_on_task(task_vector, args, model_name="1.5"):
    
    print(task_vector['input_data_init'])
    Stable_Diffusion_Unet_Path=args.sd_unet_path
    Pretrained_Unet_Path=args.pretrain_unet_path
    model_id = args.sd_path
    
    if task_vector['trained'] or os.path.exists(task_vector['finetuned_unet_path']):
        print(task_vector['finetuned_unet_path'], " trained already")
    else:
        if task_vector['input_data_init']==0:
            img_generate_start_time = time.time()
            generate_input_imgs_multi_prompts(task_vector, Stable_Diffusion_Unet_Path, Pretrained_Unet_Path, args.sd_path, model_name)
            torch.cuda.empty_cache()
            print('after generate_input_imgs_multi_prompts', torch.cuda.memory_summary())
            img_generate_end_time = time.time()
            img_generate_time = img_generate_end_time-img_generate_start_time
            print('finetune image generate time', img_generate_time)
        else:
            print("input data ready")
            
        time.sleep(60)
        
        finetune_start_time = time.time()
        if model_name=="1.5":
            finetune_model(task_vector, model_id)
        elif model_name == "xl":
            finetune_model_xl(task_vector, model_id)
        finetune_end_time = time.time()
        finetune_time = finetune_end_time - finetune_start_time
        print('finetune model time', finetune_time)
    save_task_vector(Pretrained_Unet_Path, task_vector, model_name)

def model_edit(config_data, args, model_name):
    
    Stable_Diffusion_Unet_Path=args.sd_unet_path
    Pretrained_Unet_Path=args.pretrain_unet_path
    model_id = args.sd_path
    
    task_vector_applied = config_data['task_vector_applied']
    task_vectors = config_data['task_vectors']
    merge = config_data['merge']
    
    edited_unet_path, whole_task_name=init_task_vector(task_vectors, args)
    if 'edited_unet_path' in config_data:
        edited_unet_path = config_data['edited_unet_path']
    
    fresh_sd(Stable_Diffusion_Unet_Path, Pretrained_Unet_Path)

    for task_vector in task_vectors:
        finetune_on_task(task_vector, args, model_name)
    if task_vector_applied & os.path.exists(edited_unet_path):
        pass
    else:
        task_vector_apply(Pretrained_Unet_Path, edited_unet_path, task_vectors, merge, model_name)
    return edited_unet_path