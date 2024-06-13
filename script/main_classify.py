import os
import numpy as np 
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torchvision import models
from torchvision import transforms
from PIL import Image
from lib.dataset_setup import Binary
from lib.dataset_setup import My_Dataset
from lib.nets import ResNet18
from lib.utils_data import generate_imgs, make_folder
import yaml
import argparse
import json

work_dir = os.environ.get("LLMEthicsPatchHome")
parser = argparse.ArgumentParser() 
parser.add_argument("--sd_path", type=str, default=work_dir+"/stable-diffusion-v1-5", help="the home for stable diffusion path")
parser.add_argument("--sd_unet_path", type=str, default=work_dir+"/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin", help="Unet path of Stable Diffusion")
parser.add_argument("--pretrain_unet_path", type=str, default=work_dir+"/unet_backup/unet_original_diffusion_pytorch_model.bin", help="Unet Path for Pretrain")
#parser.add_argument("--data_dir", type=str, default=work_dir+"/LLMEthicsPatches/data/", help="Data dir")
#parser.add_argument("--finetuned_models_dir", type=str, default=work_dir+"/LLMEthicsPatches/models/", help="Models dir")
#parser.add_argument("--edited_models_dir", type=str, default=work_dir+"/LLMEthicsPatches/edited_models/", help="Models dir")
#parser.add_argument("--task_vectors_dir", type=str, default=work_dir+"/LLMEthicsPatches/task_vectors/", help="Task Vectors dir")
parser.add_argument("--config_yaml", type=str, default="config.yaml", help="the config file")
args = parser.parse_args()

def loss_picker(loss):
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'cross':
        criterion = nn.CrossEntropyLoss()
    else:
        print("automatically assign mse loss function to you...")
        criterion = nn.MSELoss()

    return criterion

def optimizer_picker(optimization, param, lr):
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = optim.SGD(param, lr=lr)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = optim.Adam(param, lr=lr)
    return optimizer

def train_classifier(train_data_loader, test_data_loader, epoch, batch_size, loss_mode, optimization, lr, device, save_model_path): 
    net = ResNet18()
    net = net.cuda()

    criterion = loss_picker(loss_mode)
    optimizer = optimizer_picker(optimization, net.parameters(), lr=lr)

    train_process = []
    print("### EPOCH is %d, Learning Rate is %f" % (epoch, lr))
    print("### Train set size is %d, test set size is %d\n" % (len(train_data_loader.dataset), len(test_data_loader.dataset)))
    for epo in range(epoch):
        loss = train(net, train_data_loader, criterion, optimizer, loss_mode)
        acc_train = eval(net, train_data_loader, batch_size=batch_size)
        acc_test = eval(net, test_data_loader, batch_size=batch_size)

        print("# EPOCH%d   loss: %.4f  training acc: %.4f, testing acc: %.4f\n"\
              % (epo, loss.item(), acc_train, acc_test))

    # save model 
    state = {               
            'state_dict': net.state_dict(),'acc_ori': acc_test,}
    torch.save(state, save_model_path)

def load_classifier(load_model_path):
    net = ResNet18()
    net = net.cuda()
    net.load_state_dict(torch.load(load_model_path)['state_dict'])
    return net

def make_classification(net, img_path, prompts):
    net.eval()
    # 2. 定义预处理变换
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(img_path)

    # 4. 对图像进行预处理
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to('cuda')

    # 5. 使用模型进行预测
    with torch.no_grad():
        output = net(input_batch)

    # 6. 解码预测结果
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    class_index = torch.argmax(probabilities).item()
    return prompts[class_index]
    
def train(model, data_loader, criterion, optimizer, loss_mode):
    running_loss = 0
    model.train()
    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()
        #print(batch_x.shape)
        output = model(batch_x) # get predict label of batch_x

        if loss_mode == "mse":
            loss = criterion(output, batch_y) # mse loss
        elif loss_mode == "cross":
            loss = criterion(output, torch.argmax(batch_y, dim=1)) # cross entropy loss

        loss.backward()
        optimizer.step()
        running_loss += loss
    return running_loss

def eval(model, data_loader, batch_size=64, mode='backdoor', print_perform=False):
    model.eval() # switch to eval status
    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_y_predict = model(batch_x)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_predict.append(batch_y_predict)
        
        batch_y = torch.argmax(batch_y, dim=1)
        y_true.append(batch_y)

    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)

    if print_perform and mode is not 'backdoor':
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))

    return accuracy_score(y_true.cpu(), y_predict.cpu())

def classifier_main(train_init, data_init, prompts, train_img_num, test_img_num, val_img_num, save_model_path, data_path, model_id, sd_unet_path, pretrain_unet_path):

    if data_init:
        make_folder(data_path)
        make_folder(data_path+"/train")
        make_folder(data_path+"/test")
        make_folder(data_path+"/val")
        for prompt_idx, prompt in enumerate(prompts):
            if prompt=="naked woman":
                continue
            make_folder(data_path+"/train/"+str(prompt_idx))
            generate_imgs(
                model_id=model_id, 
                sd_unet_path=sd_unet_path, 
                pretrain_unet_path=pretrain_unet_path, 
                prompt=prompt, 
                data_folder=data_path+"/train/"+str(prompt_idx), 
                img_num=train_img_num
            )
            make_folder(data_path+"/test/"+str(prompt_idx))
            generate_imgs(
                model_id=model_id, 
                sd_unet_path=sd_unet_path, 
                pretrain_unet_path=pretrain_unet_path, 
                prompt=prompt, 
                data_folder=data_path+"/test/"+str(prompt_idx), 
                img_num=test_img_num
            )
            make_folder(data_path+"/val/"+str(prompt_idx))
            generate_imgs(
                model_id=model_id, 
                sd_unet_path=sd_unet_path, 
                pretrain_unet_path=pretrain_unet_path, 
                prompt=prompt, 
                data_folder=data_path+"/val/"+str(prompt_idx), 
                img_num=val_img_num
            )
    else:
        pass
    print("data ready")
    
    train_set = Binary(train=True, root=data_path)
    test_set = Binary(train=False, root=data_path)
    
    train_data = My_Dataset(train_set, device='cuda')
    test_data = My_Dataset(test_set, device='cuda')

    train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=8, shuffle=True, drop_last=True)
    
    if train_init:
        train_classifier(train_data_loader=train_loader, test_data_loader=test_loader, epoch=10, batch_size=16, loss_mode='cross', optimization='sgd', lr=0.01, device='cuda', save_model_path=save_model_path)
    else:
        print('train already')

if __name__ == "__main__":
    with open(args.config_yaml, "r") as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    classifier_config = config_data['classifier_config']
    train_init = classifier_config['train_init']
    prompts = classifier_config['prompts']
    if train_init:
        data_init = classifier_config['data_init']
        
        train_img_num = classifier_config['train_img_num']
        test_img_num = classifier_config['test_img_num']
        val_img_num = classifier_config['val_img_num']
        save_model_path=classifier_config['save_model_path']
        data_path=classifier_config['data_path']
        model_id = args.sd_path
        sd_unet_path = args.sd_unet_path
        pretrain_unet_path = args.pretrain_unet_path
        classifier_main(train_init, data_init, prompts, train_img_num, test_img_num, val_img_num, save_model_path, data_path, model_id, sd_unet_path, pretrain_unet_path)
    #img_path=classifier_config['img_path']
    img_folder=classifier_config['img_folder']
    net = load_classifier(save_model_path)
    #predicted_label = make_classification(net, img_path)
    img_path_list=os.listdir(img_folder)
    img2label_map={}
    for img_path in img_path_list:
        absolute_img_path = img_folder+"/"+img_path
        predicted_label = make_classification(net, absolute_img_path)
        img2label_map[img_path]=predicted_label
    with open(img_folder+"/"+"result.json") as f:
        json.load(f, img2label_map)