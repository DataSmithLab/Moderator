import yaml
import os

def make_folder(folder_path):
    if not os.path.exists(folder_path):  # 检查文件夹是否存在
        os.mkdir(folder_path)  # 创建文件夹
        print(f"文件夹已创建: {folder_path}")
    else:
        print(f"文件夹已存在: {folder_path}")

with open('/root/autodl-fs/LLMEthicsPatches/configs/permission_interfaces_setting.yaml', "r") as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)
for word_type, word_list in yaml_data.items():
    #word_type_config_filename="configs/pi_demo_configs/"+str(idx)+"-"+word_type+".yaml"
    word_type_config = {}
    task_vectors_list=[]
    word_idx=0
    
    
    example_prompts = ['Horse', 'Flower', 'Ship', 'House', 'Computer']
    example_names = ['Horse', 'Flower', 'Ship', 'House', 'Computer']
    word_dict = {}
    
    word_dict['name']=word_type
    word_dict['prompt']=word_list[0]
    word_dict['operator']="-"
    word_dict['gen_img_num_per_prompt']=10
    
    for word in word_list:
        example_prompts.append(word)
        example_names.append(word.replace(" ", "_"))
    
    word_type_config['gen_img_num_per_prompt']=10
    word_type_config['example_prompts']=example_prompts
    word_type_config['example_names']=example_names
    word_type_config['example_gen_num']=40
    word_type_config['example_group_config']={'image_column': 8, 'image_row': 5, 'image_size': 512}
    word_type_config['plot_init_example']=0
    
    make_folder("configs/exp-1/{0}".format(word_type))
    
    # 1-2-input_num
    for input_num in range(100, 1100, 100):
        word_dict['input_num']=input_num
        word_dict['train_step']=1000
        word_dict['scale']=1.0
        word_dict['trained']=0
        word_dict['input_data_init']=1
        word_type_config['task_vectors']=[word_dict]
        
        word_type_config['before_folder_name'] = "/exp-1/"+"{0}-scale_{1}-train_step_{2}-img_num_{3}".format(word_type, str(1.0), str(1000), str(input_num))
        word_type_config['after_folder_name'] = "/exp-1/"+"{0}-scale_{1}-train_step_{2}-img_num_{3}".format(word_type, str(1.0), str(1000), str(input_num))
        
        word_type_config_filename="configs/exp-1/{0}/{0}-scale_{1}-train_step_{2}-img_num_{3}.yaml".format(word_type, str(1.0), str(1000), str(input_num))
        with open(word_type_config_filename,"w",encoding="utf-8") as f:
            yaml.dump(word_type_config,f, allow_unicode=True)
    # 1-1-scales
    for scale_int in range(1, 21):
        scale = float(scale_int/10)
        word_dict['input_num']=100
        word_dict['train_step']=1000
        word_dict['scale']=scale
        word_dict['trained']=1
        word_dict['input_data_init']=1
        word_type_config['task_vectors']=[word_dict]
        
        word_type_config['before_folder_name'] = "/exp-1/"+"{0}-scale_{1}-train_step_{2}-img_num_{3}".format(word_type, str(scale), str(1000), str(100))
        word_type_config['after_folder_name'] = "/exp-1/"+"{0}-scale_{1}-train_step_{2}-img_num_{3}".format(word_type, str(scale), str(1000), str(100))        
        
        word_type_config_filename="configs/exp-1/{0}/{0}-scale_{1}-train_step_{2}-img_num_{3}.yaml".format(word_type, str(scale), str(1000), str(100))
        with open(word_type_config_filename,"w",encoding="utf-8") as f:
            yaml.dump(word_type_config,f, allow_unicode=True)
    # 1-3-train_step
    for train_step in range(100, 2100, 100):
        word_dict['input_num']=100
        word_dict['train_step']=train_step
        word_dict['scale']=1.0
        word_dict['trained']=0
        word_dict['input_data_init']=1
        word_type_config['task_vectors']=[word_dict]
                
        word_type_config['before_folder_name'] = "/exp-1/"+"{0}-scale_{1}-train_step_{2}-img_num_{3}".format(word_type, str(1.0), str(train_step), str(100))
        word_type_config['after_folder_name'] = "/exp-1/"+"{0}-scale_{1}-train_step_{2}-img_num_{3}".format(word_type, str(1.0), str(train_step), str(100))     
        
        word_type_config_filename="configs/exp-1/{0}/{0}-scale_{1}-train_step_{2}-img_num_{3}.yaml".format(word_type, str(1.0), str(train_step), str(100))
        with open(word_type_config_filename,"w",encoding="utf-8") as f:
            yaml.dump(word_type_config,f, allow_unicode=True)
    
    word_dict['input_num']=100
    word_dict['train_step']=1000
    word_dict['scale']=1.0
    word_dict['trained']=0
    word_dict['input_data_init']=1
    word_type_config['task_vectors']=[word_dict]
    word_type_config['before_folder_name'] = "/exp-1/"+"{0}-scale_{1}-train_step_{2}-img_num_{3}".format(word_type, str(1.0), str(1000), str(100))
    word_type_config['after_folder_name'] = "/exp-1/"+"{0}-scale_{1}-train_step_{2}-img_num_{3}".format(word_type, str(1.0), str(1000), str(100))   
    
    word_type_config_filename="configs/exp-1/{0}/{0}-scale_{1}-train_step_{2}-img_num_{3}.yaml".format(word_type, str(1.0), str(1000), str(100))
    with open(word_type_config_filename,"w",encoding="utf-8") as f:
        yaml.dump(word_type_config,f, allow_unicode=True)