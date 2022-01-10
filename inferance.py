'''
    请大家自行实现测试代码，注意提交格式
'''


import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

import torch.optim
import torchvision
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
from dataset import Food_LT
from model import resnet34
from newmodel import ResNet152
import config as cfg
from utils import adjust_learning_rate, save_checkpoint, train, validate, logger


class LT_Dataset_TEST(Dataset):
    num_classes = cfg.num_classes

    def __init__(self, cls_path, transform=None):
        self.img_path = []
        self.transform = transform

        files = os.listdir(cls_path)
        for file in files:
            self.img_path.append(cls_path + os.sep + file)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        path = self.img_path[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, os.path.basename(path)

PATH = './ckpt/model_best.pth.tar'
def main():
    device = torch.device(cfg.gpu)
    model = ResNet152()
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['state_dict_model'])
    model.to(device)
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])


    cls_path = './data/food/test'
    eval_dataset = LT_Dataset_TEST(cls_path=cls_path, transform=transform_test) 
    test_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=4, pin_memory=True)

    confidence = np.array([])
    pred_class = np.array([])
    predicted_class = np.array([])
    file_list = np.array([])
    with torch.no_grad():
        for i, (images,filename) in enumerate(test_loader):
            if cfg.gpu is not None:
                images = images.cuda(cfg.gpu, non_blocking=True)
            file_list = np.append(file_list,np.array(filename))
            output = model(images)
            _, predicted = output.max(1)
            predicted_class = np.append(predicted_class, predicted.cpu().numpy())    
            prob = torch.softmax(output, dim=1)
            confidence_part, pred_class_part = torch.max(prob, dim=1)
            confidence = np.append(confidence, confidence_part.cpu().numpy())
            pred_class = np.append(pred_class, pred_class_part.cpu().numpy())    
    with open("submission.csv",'w+') as f:
        f.write("Id,Expected\n")
        for id,expected in zip(file_list,predicted_class):
            f.write("{},{}\n".format(id,int(expected)))

    
if __name__ == '__main__':
    main()