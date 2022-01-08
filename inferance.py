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

from dataset import Food_LT
from model import resnet34
import config as cfg
from utils import adjust_learning_rate, save_checkpoint, train, validate, logger


PATH = './ckpt/model_best.pth.tar'
def main():
    device = torch.device("cuda")
    model = resnet34()
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()



if __name__ == '__main__':
    main()