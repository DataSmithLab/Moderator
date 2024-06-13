import yaml

INPUT_NUM=50
TRAIN_STEP=500

def add_content_tv(image_content, label_content, tv_name):
    task_vector = {}
    task_vector['name']="SP-"+str(tv_name)
    task_vector['images_configs']=[{
        "image_name":str(tv_name),
        "label_prompt":label_content,
        "real_prompt":image_content
    }]
    task_vector['operator']='+'
    task_vector['input_num']=INPUT_NUM
    task_vector['scale']=1.0
    task_vector['train_step']=TRAIN_STEP
    return task_vector

def neg_content_tv(image_content, label_content, tv_name):
    assert image_content == label_content
    task_vector = {}
    task_vector['name']="SP-"+str(tv_name)
    task_vector['images_configs']=[{
        "image_name":str(tv_name),
        "label_prompt":label_content,
        "real_prompt":image_content
    }]
    task_vector['operator']='-'
    task_vector['input_num']=INPUT_NUM
    task_vector['scale']=1.0
    task_vector['train_step']=TRAIN_STEP
    return task_vector

def replace_tvs(src_content, dst_content, src_tv_name, dst_tv_name):
    neg_src_tv = neg_content_tv(
        image_content = src_content,
        label_content = src_content,
        tv_name = src_tv_name
    )
    add_dst_tv = add_content_tv(
        image_content = dst_content,
        label_content = src_content,
        tv_name = dst_tv_name
    )
    return [neg_src_tv, add_dst_tv]

def neg_tvs(src_content=None, father_content=None, src_tv_name=None, father_tv_name=None):
    neg_src_tv = neg_content_tv(
        image_content = src_content,
        label_content = src_content,
        tv_name = src_tv_name
    )
    if father_content is None or father_content!="nan":
        return [neg_src_tv]
    else:
        add_father_tv = add_content_tv(
            image_content = father_content,
            label_content = father_content,
            tv_name = father_tv_name
        )
        return [neg_src_tv, add_father_tv]

def exp_config_gen(
        src_content,
        dst_content,
        src_name,
        dst_name,
        task_name,
        plot_img_contents,
        method="replace"
    ):
    task_name = task_name+"-"+method
    exp_config = {}
    exp_config['edited_unet_path'] = "/root/autodl-fs/LLMEthicsPatches/files/models_edited/"+"sp_example-"+str(task_name)+".safetensors"
    exp_config['merge']= False
    exp_config['task_vector_applied'] = 0
    exp_config['model_name'] = "xl"
    
    src_tv_name = src_name
    dst_tv_name = dst_name
    
    if method=="replace":
        task_vectors = replace_tvs(src_content, dst_content, src_tv_name, dst_tv_name)
    elif method == "block":
        task_vectors = neg_tvs(src_content, dst_content, src_tv_name, dst_tv_name)
    exp_config["task_vectors"] = task_vectors
    
    with open("/root/autodl-fs/LLMEthicsPatches/configs/"+"sp_example-"+str(task_name)+"-edit.yaml", "w") as f:
        yaml.dump(exp_config, f)
        
    img_config = {}
    img_config['model_name'] = "xl"
    img_config['folder_name'] = "/sp_example/"+"sp_example-"+str(src_name)
    img_config['img_gen_num'] = 20
    img_config['img_names'] = [src_name]*len(plot_img_contents)
    img_config['img_prompts'] = plot_img_contents
    img_config['gen_img_num_per_prompt']=1
    img_config['unet_path']="/root/autodl-fs/LLMEthicsPatches/files/models_edited/"+"sp_example-"+str(task_name)+".safetensors"
    with open("/root/autodl-fs/LLMEthicsPatches/configs/"+"sp_example-"+str(task_name)+"-img.yaml", "w") as f:
        yaml.dump(img_config, f)    
    return "/root/autodl-fs/LLMEthicsPatches/configs/"+"sp_example-"+str(task_name)+"-edit.yaml", "/root/autodl-fs/LLMEthicsPatches/configs/"+"sp_example-"+str(task_name)+"-img.yaml", img_config['folder_name']