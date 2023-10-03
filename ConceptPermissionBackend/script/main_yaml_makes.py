import yaml
with open('/root/autodl-fs/LLMEthicsPatches/configs/permission_interfaces_setting.yaml', "r") as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)
idx = 1
for word_type, word_list in yaml_data.items():
    word_type_config_filename="configs/pi_demo_configs/"+str(idx)+"-"+word_type+".yaml"
    word_type_config = {}
    task_vectors_list=[]
    word_idx=0
    
    example_prompts = []
    example_names = []
    
    
    for word in word_list:
        word_idx+=1
        word_dict = {}
        word_name=word.replace(" ", "_")
        word_name=str(idx)+"-"+word_type+"-"+str(word_idx)+"-"+word_name
        word_dict['name']=word_name
        word_dict['prompt']=word
        word_dict['input_num']=100
        word_dict['train_step']=1000
        word_dict['scale']=0.2
        word_dict['operator']="-"
        word_dict['trained']=1
        task_vectors_list.append(word_dict)
        
        example_prompts.append(word)
        example_names.append(word.replace(" ", "_"))
        
    word_type_config['task_vectors']=task_vectors_list
    word_type_config['example_prompts']=example_prompts
    word_type_config['example_names']=example_names
    word_type_config['example_gen_num']=16
    word_type_config['example_group_config']={'image_column': 4, 'image_row': 4, 'image_size': 512}
    word_type_config['plot_init_example']=1
    with open(word_type_config_filename,"w",encoding="utf-8") as f:
        yaml.dump(word_type_config,f, allow_unicode=True)
    idx += 1