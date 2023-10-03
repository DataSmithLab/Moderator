import os.path as osp
from collections import defaultdict as dd
import os
import numpy as np
from tqdm import tqdm
from torchvision.datasets.folder import ImageFolder
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import ImageFilter
import torch

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class Binary(ImageFolder):
    def __init__(self, train=True, transform=None, target_transform=None, root=None):
        #root = './data/generate/'
        root = osp.join(root, 'train') if train else osp.join(root, 'test')
        
        if train is True:
            flag = 'train'
        else:
            flag = 'test'
        # Initialize ImageFolder
        print('Load Data in:', root, '... ...')
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        # Use 'train' folder as train set and 'val' folder as test set

        self.data, self.targets = self.load_img_and_labels(train=flag)

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))

    def load_img_and_labels(self, train='train'):
        import torch
        print(train)
        if os.path.exists(os.path.join(self.root + '.pth')):
            imgs = torch.load(os.path.join(self.root + '.pth'))
        else:
            imgs = []
            for i in tqdm(range(len(self))):  # len(self)
                temp = self[i][0]
                temp = temp.resize((224, 224))
                temp = np.array(temp)
                imgs.append(temp)
            imgs = np.array(imgs)
            # print(imgs.shape)
            self.targets = np.array(self.targets)
            # print(self.targets.shape)
            torch.save(imgs, os.path.join(self.root +'.pth'))
        return imgs, self.targets


class My_Dataset(Dataset):
    def __init__(self, dataset, device=torch.device("cuda")):
        self.class_num = len(dataset.classes)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.device = device

        self.transform = transform
        self.data, self.targets = dataset.data, dataset.targets
        
        self.channels, self.width, self.height = self.__shape_info__()


    def __getitem__(self, item):
        #print(len(self.data), len(self.targets))
        img = Image.fromarray(self.data[item], mode='RGB')
        label_idx = self.targets[item]
        label = np.zeros(2)
        label[label_idx] = 1  # change label to one-hot list
        label = torch.Tensor(label)
        img = self.transform(img)
        img = img.to(self.device)
        label = label.to(self.device)
        return img, label

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[1:]