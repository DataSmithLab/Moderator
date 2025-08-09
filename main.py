from moderator.src.moderator_manager import ModeratorManager
from moderator.src.configs.moderator_config import ModeratorConfig
from moderator.src.configs.experiment_config import PolicyConfig, build_policy_config_from_dict
from moderator.src.configs.image_config import ImageConfig
import argparse
import json

class ModeratorArgPaser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--model_name", type=str, default="sdxl")
        self.parser.add_argument("--work_dir", type=str, default="sdxl")
        self.parser.add_argument("--exp_config_filepath", type=str, default="exp_config.json")
        self.args = self.parser.parse_args()

    def get_args(self):
        return self.args

if __name__ == "__main__":
    parser = ModeratorArgPaser()
    args = parser.get_args()

    moderator_config = ModeratorConfig(
        model_name=args.model_name,
        work_dir=args.work_dir
    )
    moderator_manager = ModeratorManager(moderator_config)

    exp_config = json.load(open(args.exp_config_filepath, "r"))
    task_type = exp_config["task_type"]
    if task_type == "generate_pretrain_image":
        image_config = ImageConfig(
            moderator_config,
            exp_config
        )
        folder_path = moderator_manager.generate_pretrain_image(
            image_config
        )
    elif task_type == "generate_image":
        image_config = ImageConfig(
            moderator_config,
            exp_config
        )
        folder_path = moderator_manager.generate_image(
            image_config
        )
    elif task_type == "model_edit":
        policy_config = build_policy_config_from_dict(
            exp_config,
            moderator_config
        )    
        folder_path = moderator_manager.edit_model(
            policy_config
        )
