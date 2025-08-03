from moderator.src.moderator_config import ModeratorConfig
from moderator.src.experiment_config import ExperimentConfig
from moderator.src.context_desc import ContextDesc
import os

task_name = "test"
src_content = ContextDesc(
    obj="cat"
)
dst_content = ContextDesc(
    obj="dog"
)
model_name = "sdxl"
method = "replace"

moderator_config = ModeratorConfig(
    model_name=model_name,
    work_dir=os.environ.get("ModeratorWorkDir")
)

experiment_config = ExperimentConfig(
    task_name=task_name,
    src_content=src_content,
    dst_content=dst_content,
    method=method,
    moderator_config=moderator_config
)

print(
    experiment_config
)