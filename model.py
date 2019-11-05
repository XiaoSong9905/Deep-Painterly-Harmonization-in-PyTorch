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

vgg19_dict = [
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2','pool1', 
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    'conv5_1',  'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
]
    
# TODO add model dict for resnet

def build_backbone(cfg, device):
    donwload_weight = True
    user_pretrained_dict = None
    if cfg.model_file is not None:
        user_pretrained_dict = torch.load(cfg.model_file)
        donwload_weight = False

    if cfg.model == 'vgg16':
        net, layer_list = models.vgg16(pretrained=donwload_weight).features, vgg16_dict
    elif cfg.model == 'vgg19':
        net, layer_list = models.vgg19(pretrained=donwload_weight).features, vgg19_dict 
    else:
        # TODO change to raise error 
        return None 
    
    if user_pretrained_dict is not None:
        net_state_dict = net.state_dict()
        user_pretrained_dict = {k: v for k, v in user_pretrained_dict.items() if k in net_state_dict}
        # TODO need to raise information for which layer weigth have been added 
        net_state_dict.update(user_pretrained_dict)
        net.load_state_dict(net_state_dict)

    net = net.to(device).eval() # freeze parameter, only use the feature part not classifier part 
    for param in net.parameters(): # set require grad to be false to save memory when capturing content and style 
        param.requires_grad = False

    # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/3 
    # TODO not sure if gradient will be backprop to image layer, need to check neural-style imlementation 

    #if cfg.debug_mode:
    #    print('===>build model')
    #    print('cnn backbone is', net)

    assert(net.training == False)
    assert(len(net) == len(layer_list))

    return net, layer_list

class ContentLoss(nn.Module):
    def __init__(self, content_weight, layer_mask):
        super(ContentLoss, self).__init__()
        self.weight = content_weight
        self.criterian = nn.MSELoss()
        self.mask = layer_mask.clone()
        self.content_fm = None # Store the original naive stich image's feature map 
        self.mode = 'None'
        self.loss = None 
    
    def forward(self, input):
        # TODO check what the capture mode have really capture 
        if self.mode == 'capture':
            self.content_fm = input.detach()
            print('ContentLoss content feature map with shape {} captured'.format(str(self.content_fm.shape)))
        elif self.mode == 'loss':
            self.loss = self.criterian(input, self.target) * self.weight
        
        # If None, do nothing 
        return input

class StyleLossPass1(nn.Module):
    '''
    Process:
        1. pass style image and record style image feature map 
        2. pass naive stich image and record content image feature map 
            (we use naive stich image and content image interchanable in this project)
        3. use two saved feature map compute match for patch inside mask on style image 
        4. compute gramma matrix for the matched 3D matrix (from style image) and save 
            4.1 this gramm matrix only correspond to the masked region 
        5. During iteration, compute gramma matrix for img for areas under mask 
        6. record the loss between two gramm matrix and output the input (since the loss is build as part of the network)
        7. use closure() for optimizer to cumulate all the loss and do loss.backward()
    '''
    def __init__(self, style_weight, layer_mask, match_patch_size):
        super(StyleLossPass1, self).__init__()
        self.weight = style_weight
        self.critertain = nn.MSELoss()
        self.mask = layer_mask.clone()
        print('StyleLossPass1 mask with shape {} registered'.format(str(self.mask.shape)))
        self.style_fm = None # Store the styled image's feature map C * H * W 
        self.content_fm = None # Store the naive stich image's feature map 
        self.target_gram = None # gram matrix of style image under the mask constrain 
        self.mode = 'None'
        self.loss = None 
        self.match_patch_size = match_patch_size

    def forward(self, input):
        if self.mode == 'capture_style':
            self.style_fm = input.detach()
            print('StyleLossPass1 style feature map with shape {} captured'.format(str(self.style_fm.shape)))
        elif self.mode == 'capture_content':
            self.content_fm = input.detach()
            print('StyleLossPass1 content feature map with shape {} captured'.format(str(self.content_fm.shape)))
        elif self.mode == 'loss':
            # TODO complete this function 
            # compute gram with input (content image)
            # use input gram and target gram to compute loss 
            return input
        # If None, do nothing 
        return input

    def compute_match(self):
        '''
        Process:
            Instead of only compute the match for pixel inside the mask, here we compute the 
                match for the whole feature map and use mask to filter out the matched pixel we don't need 
            For feature map of size C * H * W, one unit of patch is by default set to be C * 3 * 3 and L2 
                Norm is computed on top of it.
            Patch Match is done in convolution fashion where content image's patch is used as kernal and applied 
                to the styled image, a score map will be computed and we construct the matched style feature map 
                on top of the score map 
            Content Feature Map is set to be unchanged and we split the Style Feature Map and create a new 
                matched style feature map 
        '''
        assert(self.content_fm is not None)
        assert(self.style_fm is not None)

        # Create a copy of style image fm (to matain size)
        self.match_style_fm = self.style_fm.clone() # create a copt of the same size 
        n_patch_h = math.floor(self.match_style_fm.shape[1] / self.match_patch_size) # use math package to avoid potential python2 issue 
        n_patch_w = math.floor(self.match_style_fm.shape[0] / self.match_patch_size)

        # TODO this function have not yet been tested yet
        # Use content image feature map as kernal and compute score map on style image
        stride = self.match_patch_size
        for i in range(n_patch_h):
            for j in range(n_patch_w):
                kernal = self.content_fm[:, :, i*self.match_patch_size:i*(self.match_patch_size + 1), \
                    j*self.match_patch_size:j*(self.match_patch_size+1)]
                print('kernal size', kernal.shape)
                print('fm size', self.match_style_fm.shape)

                # For F.conv2d : https://pytorch.org/docs/stable/nn.functional.html
                score_map = F.conv2d(self.style_fm, kernal, stride=stride)
                
                idx = torch.argmax(score_map).item()
                indices = (int(idx / score_map.shape[1]), int(idx % score_map.shape[1]))
                
                self.match_style_fm[:, :, i*self.match_patch_size:i*(self.match_patch_size + 1), \
                    j*self.match_patch_size:j*(self.match_patch_size+1)] = self.style_fm[:, :, \
                        indices[0]*self.match_patch_size: indices[0]*(self.match_patch_size+1), \
                            indices[1]*self.match_patch_size: indices[1]*(self.match_patch_size+1)]

        return None

    def compute_style_gramm(self, fm3d):
        # TODO finish this function 

        gram = None 
        return gram 

class StyleLossPass2(StyleLossPass1):
    '''
    child class of StyleLossPass1 that's capable of compute nearest neighbor like pass 1 
    '''
    def __init__(self):
        super(StyleLossPass2, self).__init__()

    def forward(self):
        return None 