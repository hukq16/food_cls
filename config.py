root = '/home/whutyw/food_cls'
log_path = './log/log.txt'
resume = None

gpu = 0
num_classes = 101
lr = 0.01
batch_size = 64
weight_decay = 2e-4
num_epochs = 200
momentum = 0.9
cos = False
weighted_sample=False

# loss
use_focal = False
use_weighted_ce = False

# augmentation
mixup = False
cutmix_prob = 0.5

# model
use_vit = False
use_efficient = False
use_se = False