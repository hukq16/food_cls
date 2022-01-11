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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import Food_LT
from model import resnet34
from resmodel import ResNet152
from newmodel import efficientnet_b7
from newmodel import se_resnext_152
import config as cfg
from utils import adjust_learning_rate, save_checkpoint, train, validate, logger
from focalLoss import create_focal_loss
from WeightedSoftmaxLoss import create_weighted_loss
from vit import vit_s

def main():
    if cfg.use_vit:
        model = vit_s()
    elif cfg.use_efficient:
        model = efficientnet_b7()
    elif cfg.use_se:
        model = se_resnext_152()
    else:
        model = ResNet152()
    
    if cfg.resume:
        ''' plz implement the resume code by ur self! '''
        pass

    print('log save at:' + cfg.log_path)
    logger('', init=True)
    
    if not torch.cuda.is_available():
        logger('Plz train on cuda !')
        os._exit(0)

    if cfg.gpu is not None:
        print('Use cuda !')
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)

    print('Load dataset ...')
    dataset = Food_LT(False, root=cfg.root, batch_size=cfg.batch_size, num_works=4)

    train_loader = dataset.train_instance
    val_loader = dataset.eval
    
    if cfg.use_focal:
        criterion = create_focal_loss().cuda(cfg.gpu)
    elif cfg.use_weighted_ce:
        criterion = create_weighted_loss().cuda(cfg.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(cfg.gpu)
    if cfg.use_vit:
        optimizer = torch.optim.Adam([{"params": model.parameters()}], lr = 3e-4)
    else:
        optimizer = torch.optim.SGD([{"params": model.parameters()}], cfg.lr,
                                    momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    
    if cfg.use_se:
        scheduler_1 = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3,verbose=True)
    best_acc = 0
    for epoch in range(cfg.num_epochs):
        logger('--'*10 + f'epoch: {epoch}' + '--'*10)
        logger('Training start ...')
        
        if not cfg.use_se:
            adjust_learning_rate(optimizer, epoch, cfg)
        
        train(train_loader, model, criterion, optimizer, epoch)
        logger('Wait for validation ...')
        acc = validate(val_loader, model, criterion)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        logger('* Best Prec@1: %.3f%%' % (best_acc))
        
        if cfg.use_se:
            scheduler_1.step(acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict_model': model.state_dict(),
            'best_acc': best_acc,
        }, is_best, cfg.root)

    print('Finish !')


if __name__ == '__main__':
    main()