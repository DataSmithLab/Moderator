import subprocess
import yaml
import os

all_task_vectors = []
config_filenames=os.listdir("/root/autodl-fs/LLMEthicsPatches/configs/permission_interfaces_configs")
for config_filename in config_filenames:
    if ".yaml" in config_filename:
        with open("/root/autodl-fs/LLMEthicsPatches/configs/permission_interfaces_configs/"+config_filename, "r") as yaml_file:
            config_data = yaml.safe_load(yaml_file)
        task_vectors = config_data["task_vectors"]
        all_task_vectors += task_vectors

record_filename="similarity.json"
failed_tasks = []
print('task vectors number:', len(all_task_vectors))
for task_vector_A in all_task_vectors:
    for task_vector_B in all_task_vectors:
        A_name=task_vector_A['prompt']
        B_name=task_vector_B['prompt']
        A_path='/root/autodl-fs/LLMEthicsPatches/task_vectors/'+task_vector_A['name']+'-1000-1.0.npy'
        B_path='/root/autodl-fs/LLMEthicsPatches/task_vectors/'+task_vector_B['name']+'-1000-1.0.npy'
        
        script_path = "main_cosine.py"
        args = [
            "--A_name", A_name,
            "--B_name", B_name,
            "--A_path", A_path,
            "--B_path", B_path,
            "--record_filename", record_filename
        ]
        command = ["python", script_path] + args
        command_str = " ".join(args)
        #print(command_str)
        try:
            subprocess.run(command, check=True)
            #print(A_name, B_name)
        except:
            failed_tasks.append(A_name+", "+B_name)
print(failed_tasks)