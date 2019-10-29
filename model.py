# EECS 442 @ UMich Final Project 
# No commercial Use Allowed 

import os 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torchvision.transforms as transforms 
import torchvision.models as models
import torchvision
import torch.nn.functional as F 
from PIL import Image 
import argparse
import copy
import math 
import numpy as np 
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

vgg16_dict = [
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2','pool1', 
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
    'conv5_1',  'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5'
    ]
    
# TODO add model dict for resnet

def build_backbone(cfg, device):
    donwload_weight = True
    user_pretrained_dict = None
    if cfg.model_file is not None:
        user_pretrained_dict = torch.load(cfg.model_file)
        donwload_weight = False

    if cfg.model == 'vgg16':
        net, layer_list = models.vgg16(pretrained=donwload_weight), vgg16_dict
    elif cfg.model == 'vgg19':
        net, layer_list = models.vgg19(pretrained=donwload_weight), vgg19_dict
    elif cfg.model == 'resnet18':
        net, layer_list = models.resnet18(pretrained=donwload_weight), None 
    elif cfg.model == 'resnet34':
        net, layer_list = models.resnet34(pretrained=donwload_weight), None 
    else:
        # TODO chaneg to raise error 
        return None 
    
    if user_pretrained_dict is not None:
        net_state_dict = net.state_dict()
        user_pretrained_dict = {k: v for k, v in user_pretrained_dict.items() if k in net_state_dict}
        # TODO need to raise information for which layer weigth have been added 
        net_state_dict.update(user_pretrained_dict)
        net.load_state_dict(net_state_dict)

    net = net.features.to(device).eval() # freeze parameter, only use the feature part not classifier part 
    for param in net.parameters(): # set require grad to be false to save memory when capturing content and style 
        param.requires_grad = False

    # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/3 
    # TODO not sure if gradient will be backprop to image layer, need to check neural-style imlementation 

    if cfg.debug_mode:
        print('===>build model')
        print('cnn backbone is', net)

    assert(net.training == False)
    assert(len(net) == len(layer_list))

    return net, layer_list

class ContentLoss(nn.Module):
    def __init__(self, content_weight, layer_mask):
        super(ContentLoss, self).__init__()
        self.weight = content_weight
        self.criterian = nn.MSELoss()
        self.mask = layer_mask.clone()
        self.target = None # Store the original naive stich image's feature map 
        self.mode = 'None'
        self.loss = None 
    
    def forward(self, input):
        # TODO check what the capture mode have really capture 
        if self.mode == 'capture':
            self.target = input.detach()
        elif self.mode == 'loss':
            self.loss = self.criterian(input, self.target) * self.weight
        else:
            # If None, do nothing 
        return input

class StyleLossPass1(nn.Module):
    def __init__(self, style_weight, layer_mask):
        super(StyleLossPass1, self).__init__()
        self.weight = style_weight
        self.critertain = None # Should be gramm matric 
        self.mask = layer_mask.clone()
        self.target = None # Store the styled image's feature map 
        self.mode = 'None'
        self.loss = None 

    def forward(self, input):
        if self.mode == 'capture':
            self.target = input.detach()
        elif self.mode == 'loss':
            self.loss = self.critertain(input, self.target, self.mask)
        else:
           # If None, do nothing 
        return input

class StyleLossPass2(nn.Module):
    def __init__(self):
        super(StyleLossPass2, self).__init__()
    def forward():
        return None 