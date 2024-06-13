import os

def fresh_sd(sd_unet_path, unet_path):
    try:
        os.remove(sd_unet_path)
    except:
        pass
    os.symlink(unet_path, sd_unet_path)
    
def init_task_vector(task_vectors, args):
    unet_file_name="unet_"
    whole_task_name="sd_"
    for idx in range(len(task_vectors)):
        task_vectors[idx]['input_data_dir']=args.data_dir+"/input-"+task_vectors[idx]['name']+"-input_num-"+str(task_vectors[idx]['input_num'])
        task_vectors[idx]['finetuned_model_dir']=args.finetuned_models_dir+"finetuned_Unet-"+task_vectors[idx]['name']+"-train_step-"+str(task_vectors[idx]['train_step'])+"-input_num-"+str(task_vectors[idx]['input_num'])
        task_vectors[idx]['finetuned_unet_path']=args.finetuned_models_dir+"finetuned_Unet-"+task_vectors[idx]['name']+"-train_step-"+str(task_vectors[idx]['train_step'])+"-input_num-"+str(task_vectors[idx]['input_num'])+"/unet/diffusion_pytorch_model.bin"
        task_vectors[idx]['task_vector_path']=args.task_vectors_dir+task_vectors[idx]['name']+"-"+str(task_vectors[idx]['train_step'])+"-"+str(task_vectors[idx]['scale'])+".npy"
        task_vectors[idx]['full_task_vector_path']=args.task_vectors_dir+task_vectors[idx]['name']+"-train_step-"+str(task_vectors[idx]['train_step'])+"-operator-"+str(task_vectors[idx]['operator'])+"-scale-"+str(task_vectors[idx]['scale'])+".npy"
        unet_file_name+="'"+task_vectors[idx]["operator"]+task_vectors[idx]['name']+"_"+str(task_vectors[idx]['scale'])+"'"
        whole_task_name+="'"+task_vectors[idx]["operator"]+task_vectors[idx]['name']+"_"+str(task_vectors[idx]['scale'])+"'"
    unet_file_name+=".bin"
    edited_unet_path=args.edited_models_dir
    edited_unet_path += unet_file_name
    return edited_unet_path, whole_task_name