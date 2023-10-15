#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import optuna  
import collections
import math
import os
import shutil
import pandas as pd
import torch
from torch.utils import data
import torchvision
from torchvision.datasets.folder import *
from torch import nn
import torch.nn.functional as F
from d2l import torch as d2l
import pandas as pd
import timm
import time
from torchinfo import summary
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import numpy as np
from datetime import datetime
from prettytable import PrettyTable


# In[2]:


config = {
    'img_size_h': 224,
    'img_size_w': 224,
    'min_lr': 1e-8,
    'num_workers': 8,
    'conv_A': 'resnet18',
    'conv_B': 'resnet18',
    'batch_size': 22,
    'pretrained': False,
    'dropout': 0.5,
    'number_of_class': 7
    }


# In[1]:


rootdir = 'your_data_root'
classes = os.listdir(os.path.join(rootdir,'vision'))
all_vision_data, all_AE_data = [pd.read_csv(os.path.join(rootdir,'csvFile',m,'all.csv')) for m in ('vision','AE')]
train_vision_path, train_AE_path = [os.path.join(rootdir,'csvFile',m,'train.csv') for m in ('vision','AE')]
valid_vision_path, valid_AE_path = [os.path.join(rootdir,'csvFile',m,'valid.csv') for m in ('vision','AE')]
test_vision_path, test_AE_path = [os.path.join(rootdir,'csvFile',m,'test.csv') for m in ('vision','AE')]


# In[5]:


transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize((config['img_size_h'], config['img_size_w'])),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]),#会影响可视化结果
    ])

transform_valid = torchvision.transforms.Compose([
    torchvision.transforms.Resize((config['img_size_h'], config['img_size_w'])),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]),
    ])
    
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((config['img_size_h'], config['img_size_w'])),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]),
    ])


# In[6]:


class ImagePairDataset(data.Dataset):

    def __init__(self, image1_path, image2_path, image_label, transform=None):
        super(ImagePairDataset, self).__init__()
        self.image1_path = image1_path
        self.image2_path = image2_path
        self.image_label = image_label
        self.classes = sorted(entry for entry in list(set(image_label)))
        self.transform = transform
 
    def __getitem__(self, index):
        image1 = Image.open(self.image1_path[index])
        image1 = image1.convert('RGB')
        image2 = Image.open(self.image2_path[index])
        image2 = image2.convert('RGB')
        label = self.image_label[index]
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
 
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return (image1, image2), class_to_idx[label]
 
    def __len__(self):
        return len(self.image1_path)


# In[8]:


def get_path_label(path):
    data = pd.read_csv(path)
    return data['path'].to_list() ,data['label'].to_list()


# In[ ]:





# In[17]:


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class self_attention_block(nn.Module):
    def __init__(self, num_hiddens, norm_shape, num_heads ,dropout, **kwargs):
        super(self_attention_block, self).__init__(**kwargs)
        self.self_attention = nn.MultiheadAttention(num_hiddens, num_heads, dropout=dropout)
        self.addnorm = AddNorm(norm_shape, dropout)

    def forward(self, X):
        x,x_w = self.self_attention(X, X, X)
        Y = self.addnorm(X, x)
        return Y


class Cnn_Transformer_Concat(nn.Module):
    '''Multi_Fusion''' 
    def __init__(self, num_hiddens, norm_shape, num_heads, dropout, use_bias=False,num_layers=3,**kwargs):
        super(Cnn_Transformer_Concat, self).__init__()
        self.num_hiddens = num_hiddens
        self.norm_shape = norm_shape
        self.num_heads = num_heads
        self.dropout = dropout
        self.module1 = nn.Sequential(*list(timm.create_model(config['conv_A'], 
                                        drop_rate = dropout, 
                                        pretrained=config['pretrained']).children())[:-4],
                                        nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU())
        self.module2 = nn.Sequential(*list(timm.create_model(config['conv_B'], 
                                        drop_rate = dropout, 
                                        pretrained=config['pretrained']).children())[:-4],
                                        nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU())
        self.addnorm = AddNorm(self.norm_shape, self.dropout)
        self.fc1 = nn.Linear(self.num_hiddens, 64)
        self.fc2 = nn.Linear(64, config['number_of_class'])
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
            self_attention_block(self.num_hiddens, self.norm_shape, self.num_heads, self.dropout))

    def forward(self, x1, x2):
        x1 = self.module1(x1)
        x2 = self.module2(x2)
        x = torch.cat((x1.view(x1.size(0),x1.size(1), -1), x2.view(x2.size(0),x2.size(1), -1)), dim=2)
        y = x.permute(0,2,1)
        y = self.blks(y)
        y = self.fc1(nn.functional.adaptive_avg_pool2d(y,output_size=(1,512)).squeeze(dim=1))
        y = self.fc2(y)
        return y
    


# In[18]:


def train_batch(net, features1,features2,labels,loss, trainer, devices):
    """Train for a minibatch with mutiple GPUs"""
    X1 = features1.to(devices[0])
    X2 = features2.to(devices[0]) 
    y = labels.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred= net(X1,X2)
    l= loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def evaluate_accuracy_gpu(net, valid_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.
    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for i, ((features1,features2), labels) in enumerate(valid_iter):
            X1 = features1.to(device[0])
            X2 = features2.to(device[0]) 
            y = labels.to(device[0])
            acc = d2l.accuracy(net(X1,X2), y)
            metric.add(acc,  d2l.size(y))
            
    return metric[0] / metric[1]


# In[19]:


def train(net, train_iter, valid_iter, test_iter, num_epochs, lr, wd, devices, name):
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    milestones = [200, 230, 260, 290, 320, 350]
    gamma = 0.7
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


    loss = nn.CrossEntropyLoss(reduction="none")
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    legend_test = ['test acc']
    animator1 = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    # animator2 = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
    #                         legend=legend_test)
    path = os.path.join('The location of the folder where you saved the data',name)
    os.makedirs(path, exist_ok=False)
    df_train = pd.DataFrame(columns=['step','metric[0]','metric[1]','metric[2]','train loss', 'train acc'])#列名
    df_train.to_csv(os.path.join(path , 'train_' + name +'.csv'),index=False)
    df_valid = pd.DataFrame(columns=['step','valid acc'])
    df_valid.to_csv(os.path.join(path , 'valid_' + name +'.csv'),index=False)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        for i, ((features1,features2), labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch(net, features1,features2, labels,loss, optimizer, devices)
            metric.add(l, acc, labels.shape[0])
            metric_save = [i, metric[0], metric[1], metric[2], metric[0] / metric[2], metric[1] / metric[2]]
            train_save = pd.DataFrame([metric_save])
            train_save.to_csv(os.path.join(path , 'train_' + name +'.csv'),
                                            mode='a',header=False,index=False)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator1.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            legend.append('valid acc')
            valid_acc = evaluate_accuracy_gpu(net, valid_iter,device=devices)
            save = [i, valid_acc]
            valid_save = pd.DataFrame([save])
            valid_save.to_csv(os.path.join(path , 'valid_' + name +'.csv'),
                                            mode='a',header=False,index=False)
            animator1.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
    #print(f'Total time',timer.sum())


# In[21]:


def create_net(num_hiddens, norm_shape, num_heads, dropout, num_layers, device):
    net = nn.DataParallel(Cnn_Transformer_Concat(num_hiddens=num_hiddens, 
                                            norm_shape=[1568,512], 
                                            num_heads=num_heads, dropout=dropout,
                                            num_layers=num_layers), 
                                            device_ids=device).to(device[0])
    return net

devices = d2l.try_all_gpus()


# In[26]:


num_epochs, lr, wd =400, 2.5e-06, 5.7e-07
num_layers = 1
dropout = 0.32
config['batch_size']=64
num_heads = 64
net = create_net(num_hiddens=512, norm_shape=[1568,512], 
                    num_heads=num_heads, dropout=dropout,
                    num_layers=num_layers, device=devices)
name = 'Training number'#

train_data_vision_path, train_data_vision_label = get_path_label(train_vision_path)
train_data_AE_path, train_data_AE_label = get_path_label(train_AE_path)

valid_data_vision_path, valid_data_vision_label = get_path_label(valid_vision_path)
valid_data_AE_path, valid_data_AE_label = get_path_label(valid_AE_path)

test_data_vision_path, test_data_vision_label = get_path_label(test_vision_path)
test_data_AE_path, test_data_AE_label = get_path_label(test_AE_path)


dataset_train = ImagePairDataset(train_data_vision_path, train_data_AE_path, 
                                    train_data_vision_label, transform=transform_train)
dataset_valid = ImagePairDataset(valid_data_vision_path, valid_data_AE_path, 
                                    valid_data_vision_label, transform=transform_valid)
dataset_test = ImagePairDataset(test_data_vision_path, test_data_AE_path, 
                                    test_data_vision_label, transform=transform_test)

iter_train = data.DataLoader(dataset_train, config['batch_size'], pin_memory=True,shuffle=True, 
                            drop_last=False,num_workers=config['num_workers'])
iter_valid = data.DataLoader(dataset_valid, config['batch_size'], pin_memory=True,shuffle=True, 
                            drop_last=False,num_workers=config['num_workers'])
iter_test = data.DataLoader(dataset_test, config['batch_size'], pin_memory=True,shuffle=True, 
                            drop_last=False,num_workers=config['num_workers'])


# In[27]:


'''Training'''
time1 = datetime.now()
train(net, iter_train, iter_valid, iter_test, num_epochs, lr, wd, devices,name)
time2 = datetime.now()
time = time2-time1
path_params = os.path.join('/home/zcy/data/VAFnet/data_and_result',
                            name, name+'_params' )
torch.save(net.state_dict(),path_params )
print(f"Total time: {time}")


# In[ ]:





# In[ ]:




