##机器学习课程竞赛技术报告

#####参赛队名：huashui
#####队长：江晓湖
#####小组成员：胡克勤、李敏、王鹏勇
######项目github链接：https://github.com/hukq16/food_cls
####一、小组成员具体贡献：

######江晓湖：尝试了多角度、多种方法改进模型性能，包括CutMix、Mixup、vit、weighted sampling技术等，并且其中的CutMix数据增强方式被用在了最终的方案中；与小组成员共同探讨模型性能改进的方法与思路；作为队长，协调组内成员分工与合作。

######胡克勤：对包括Resnet、efficientnet、SE_Resnet、Resnet34、Resnet101、Resnet152等在内的各类基础模型在该数据集上的性能进行实验对比，并得出结论：不加其他trick情况下Resnet152性能最好；与小组成员共同探讨模型性能改进的方法与思路。

######李敏：负责本次项目最终技术报告的撰写；与小组成员共同探讨模型性能改进的方法与思路。

######王鹏勇：对各类损失函数进行研究，尝试将focal loss、weighted loss等损失函数应用于模型，并评估其对性能的影响；与小组成员共同探讨模型性能改进的方法与思路。

####二、整体工作内容：

######在本次任务中，我们通过多种方式优化改进模型，最终在采用Resnet152网络，CutMix数据增强方法下，经200个epoch的训练后，模型在测试集上的识别精确率由baseline的26.799%提升到30.033%。

######由于训练数据集来自随机采集的自然界的食物照片，每一类数据的样本数量极不均衡。如果采用一般的模型进行学习，其结果是在样本数量多的种类上性能良好，在样本数量少的种类上性能较差。针对这一问题，我们在baseline的基础上进行了如下五个方面的尝试和改进，并取得了不错的效果提升：

##### 1、改进backbone

###### 将原来的Res34网络，加深到Res101、Res152，训练后的模型在测试集上的识别准确率由baseline的26.799%，分别提升到27.146%、28.152%。采用se_resnet152以及effcientnet网络，也一定程度有性能提升，在同样训练100个epoch的情况下，准确率也分别提升到了27.228%、27.169%。

##### 2、重新设计损失函数

######尝试使用focal loss和weighted cross entropy作为损失函数，以增加小类数据在损失函数中的权重。

######focal loss是在cross entropy的基础上加上权重，让模型注重学习难以学习的样本，训练数据不均衡中占比较少的样本，相对放大对难分类样本的梯度，相对降低对易分类样本的梯度，可在一定程度上解决类别不均衡问题。

######Weighted cross entropy即加权交叉熵，通过一个系数描述样本在loss中的重要性。对于小数目样本，加强它对loss的贡献，对于大数目样本，减少它对loss的贡献。

##### 3、Weighted Sampler

###### 对于类不平衡的数据集，那些样本极少的类别下的数据，被模型看到的机会太小，以致模型无法从该类中学到有用的特征。通WeightedRandomSampler增强小类数据被采样的概率，可使得模型在小类数据的识别上获取性能的提升。

##### 4、More Epochs

######在本次实验中，在网络、数据增强、损失函数等方面一样的情况下，我们把epoch数由原来的100增加到200，使得模型得到了更好的性能。

##### 5、数据增强

######采用了Mixup、CutMix等方法，对数据进行增强。在取得最好性能的模型中，采用的便是CutMix的数据增强方法。

######Mixup是一种运用在计算机视觉中的对图像进行混类增强的算法，它可以将不同类之间的图像进行混合，从而扩充训练数据集。

######CutMix则是将一部分区域cut掉但不填充0像素而是随机填充训练集中的其他数据的区域像素值，分类结果按一定的比例分配。在本次任务中，相比其他数据增强手段，采用CutMix方案的模型取得了最好的效果并被用于最终模型。

| model                                    | acc on val |
| ---------------------------------------- | ---------- |
| Baseline（res34+100epoch）               | 26.799%    |
| Res101+100epoch                          | 27.146%    |
| Res152+100epoch                          | 28.152%    |
| se_resnet152+100epoch                    | 27.228%    |
| efficientnet+100epoch                    | 27.169%    |
| Res101+focal loss, 100e                  | 27.624%    |
| Res101+weight cross entropy loss, 100e   | 27.756%    |
| Res101+ weighted resample, 100e          | 25.050%    |
| Res101+Mixup, 100e                       | 26.304%    |
| Res152 + weight cross entropy loss, 200e | 29.175%    |
| ViT（patch size16+depth6）, 100e         | 18.119%    |
| Res101+CutMix, 100e                      | 28.218%    |
| Res152+focal loss, 200e                  | 28.317%    |
| Res152+Cutmix, 200e                      | 30.033%    |





