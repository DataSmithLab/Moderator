from moderator.src.configs.moderator_config import ModeratorConfig
import PIL.Image as Image
import json
import os
from moderator.src.configs.task_vector_config import TVConfig
from moderator.src.model_manager import ModelManager
import toml

class DatasetManager:
    def __init__(
        self,
        moderator_config:ModeratorConfig
    ) -> None:
        self.moderator_config = moderator_config
        self.model_name = moderator_config.model_name
    
    def make_folder(
            self, 
            folder_path:str
        ):
        if not os.path.exists(folder_path):  # 检查文件夹是否存在
            os.mkdir(folder_path)  # 创建文件夹
            print(f"Dataset Folder Created: {folder_path}")
        else:
            print(f"Dataset Folder Exists: {folder_path}")

    def dataset_metadata_generate(
            self,
            dataset_dir:str,
            image_filenames:list,
            dataset_captions:list
        ):
        output_file = open(dataset_dir+'/metadata.jsonl', 'w+')
        for image, dataset_caption in zip(image_filenames, dataset_captions):
            record_str = '{"file_name": "'+image+'", "text": "'+dataset_caption+'"}'
            print(record_str, file=output_file)
        output_file.close()  

        if self.model_name == "sdxl":
            self.dataset_adjust(dataset_dir)

    def resize_image(
            self,
            input_path:str, 
            output_path:str
        ):
        with Image.open(input_path) as img:
            resized_img = img.resize((512, 512))
            resized_img.save(output_path, format='PNG')
        return f"Image resized and saved to {output_path}"

    def convert_to_dict(
            self,
            input_str:str
        ):
        input_str = input_str.replace("{", "")
        input_str = input_str.replace("}", "")
        input_str = input_str.replace("\n", "")
        pairs = input_str.split('", "')

        result_dict = {}

        for pair in pairs:
            print(pair)
            key, value = pair.split('": "')
            result_dict[key.strip("\"")] = value.strip("\"")

        return result_dict

    def dataset_adjust(
            self,
            input_data_dir:str
        ):
        f = open(input_data_dir+"/metadata.jsonl", "r")
        meta_data = f.readlines()
        new_dict = {}
        for meta_data_line in meta_data:
            meta_data_dict = self.convert_to_dict(meta_data_line)
            img_key = meta_data_dict["file_name"]
            new_img_key = img_key.replace(" ","_")
            img_caption = meta_data_dict["text"]
            if os.path.exists(input_data_dir+"/"+new_img_key):
                pass
            else:
                os.rename(input_data_dir+"/"+img_key, input_data_dir+"/"+new_img_key)
            
            print(input_data_dir+"/"+new_img_key)
            self.resize_image(input_data_dir+"/"+new_img_key, input_data_dir+"/"+new_img_key)
            
            new_dict[new_img_key] = {'caption':img_caption, 'tags':"violence"}
        with open(input_data_dir+"/metadata.json", "w+") as f:
            json.dump(new_dict, f, indent=4)

    def generate_tv_dataset(
            self,
            task_vector_config:TVConfig, 
            model_manager:ModelManager
        ):
        image_filenames, label_prompts = model_manager.generate_images(
            sd_unet_path=self.moderator_config.pretrain_unet_path,
            image_config=task_vector_config.image_config
        )
        self.dataset_metadata_generate(
            dataset_dir=task_vector_config.image_config.folder_path,
            image_filenames=image_filenames,
            label_prompts=label_prompts
        )
        if self.model_name == "sdxl":
            self.xl_finetune_toml_make(
                task_vector_config=task_vector_config
            )

    def xl_finetune_toml_make(
            self,
            task_vector_config: TVConfig
        ):
        toml_path = task_vector_config.finetuned_model_toml_path
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
                            'image_dir': task_vector_config.image_config.folder_path,
                            'metadata_file': task_vector_config.image_config.folder_path+'/metadata.json'
                        }
                    ]
                }
            ]
        }
        with open(toml_path, 'w+') as f:
            toml.dump(toml_config, f)
        return toml_path