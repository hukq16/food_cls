import config as cfg 
import torch
import torch.nn as nn

def create_weighted_loss():
    print('Loading Weighted Softmax Loss.')
    # Imagenet_LT class distribution
    dist = [0 for _ in range(101)]
    with open(cfg.root+'/data/food/train.txt') as f:
        for line in f:
            dist[int(line.split()[1])] += 1
    num = sum(dist)
    prob = [i/num for i in dist]
    prob = torch.FloatTensor(prob)
    # normalization
    max_prob = prob.max().item()
    prob = prob / max_prob
    # class reweight
    weight = - prob.log() + 1
    print("weighted ce weights",weight)
    return nn.CrossEntropyLoss(weight=weight)
