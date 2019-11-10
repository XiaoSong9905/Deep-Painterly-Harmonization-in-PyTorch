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
import time

vgg16_dict = [
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5'
]

vgg19_dict = [
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
]


def build_backbone(cfg):
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

    net = net.eval()  # freeze parameter, only use the feature part not classifier part
    for param in net.parameters():  # set require grad to be false to save memory when capturing content and style
        param.requires_grad = False

    # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/3 

    # if cfg.debug_mode:
    #    print('===>build model')
    #    print('cnn backbone is', net)

    assert (net.training == False)
    assert (len(net) == len(layer_list))

    return net, layer_list



# Ziyan use class in order to match the format in this script. However, a function may better.
# Xiao : the reason to use a class is to able to put the TVLoss class into the network that can forward, backward by simply model(img), loss.backward()
class TVLoss(nn.Module):
    def __init__(self, tv_weight):
        super(TVLoss, self).__init__()
        self.weight = tv_weight

    def forward(self, img):
        '''
        img: pytorch variable of shape (1,C,H,W)
        returns: tv loss
        '''
        #####################
        # Formula: $L_{tv} = w_t \times \left(\sum_{c=1}^3\sum_{i=1}^{H-1}\sum_{j=1}^{W} (x_{i+1,j,c} - x_{i,j,c})^2 + \sum_{c=1}^3\sum_{i=1}^{H}\sum_{j=1}^{W - 1} (x_{i,j+1,c} - x_{i,j,c})^2\right)$
        #####################
        tmp_h = torch.sum((img[:, :, 1:, :] - img[:, :, :-1, :]) ** 2)
        tmp_w = torch.sum((img[:, :, :, 1:] - img[:, :, :, :-1]) ** 2)
        self.loss = self.weight * (tmp_h + tmp_w)
        return self.loss


class ContentLoss(nn.Module):
    def __init__(self, content_weight, layer_mask):
        super(ContentLoss, self).__init__()
        self.weight = content_weight  # content loss weight
        self.criterian = nn.MSELoss()
        self.mask = layer_mask.clone()  # a weighted mask, not binary mask. To see why check `understand mask` notebook
        self.mode = 'None'

    def forward(self, input):
        if self.mode == 'capture':
            # Capture 
            self.content_fm = input.detach()
            print('ContentLoss content feature map with shape {} captured'.format(str(self.content_fm.shape)))

            # Update Mask Size after feature map is captured 
            self.mask = self.mask.expand_as(self.content_fm)
            print('ContentLoss Mask is resize to {}'.format(str(self.mask.shape)))

        elif self.mode == 'loss':
            self.loss = self.criterian(input, self.content_fm) * self.weight

            def backward_variable_gradient_mask_hook_fn(grad):
                # Complex Module do not support `register_backward_tensor()`
                # Simple solution to mask gradient is to register hook on variable 
                return torch.mul(grad, self.mask)

            input.register_hook(backward_variable_gradient_mask_hook_fn)

        return input


# Why use class here? Ziyan and Deyang think a function should be fine.
class GramMatrix(nn.Module):
    '''
    Take Reference from https://github.com/jcjohnson/neural-style/blob/master/neural_style.lua 
    To understand how gram matrix work, checkout `understand Gram Matrix` notebook 
    ! Gram Matrix is not complicated thing, it's just covariance matrix 
    '''

    def __init__(self):
        super(GramMatrix, self).__init__()

    def forward(self, input):
        '''
        Input : 
            input (represent feature map) : B * C * H * W 
        Output : 
            gram : B * (C * C) 
                   when batch size is 1, then output shape 1 * C * C 
        '''
        B, C, H, W = input.shape
        output = torch.zeros((B, C, C))
        for i in range(B):
            fm_flat = input[i].view(C, H * W)
            output[i] = torch.mm(fm_flat, fm_flat.t())

        return output


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
        self.critertain = nn.MSELoss()  # TODO check the implementation using MSELoss and gram matrix match the one in papaer
        self.gram = GramMatrix()

        self.mask = layer_mask.clone()
        self.style_fm = None  # Store the styled image's feature map C * H * W
        self.img_fm = None  # Store the naive stich image's feature map
        self.style_gram = None  # gram matrix of style image under the mask constrain (1 * C * C)

        self.mode = 'None'
        self.loss = None
        self.match_patch_size = match_patch_size

    def forward(self, input):
        if self.mode == 'capture_style':
            # Capture Style is done after capture content 
            style_fm = input.detach()
            print('StyleLossPass1 style feature map with shape {} captured'.format(str(style_fm.shape)))

            # Compute Style Gram Matrix 
            style_fm_matched = self.match_fm(style_fm, self.img_fm)
            style_fm_matched_masked = torch.mul(style_fm_matched, self.mask)
            self.style_matched_gram = self.gram(style_fm_matched_masked) / torch.sum(
                self.mask)  # See Gatys `2.2. Style representation` to see how style loss defined over gram matrix
            print(
                'StyleLossPass1 style gram matrix under mask with shape {}'.format(str(self.style_matched_gram.shape)))

            # Delete unused variable to save memory, we only need the matched gram matrix 
            del self.img_fm
            del style_fm

        elif self.mode == 'capture_content':
            # Capture Content fm first, since we need content image fm when initialize the match during style match 
            self.img_fm = input.detach()
            print('StyleLossPass1 img (naive stich) feature map with shape {} captured'.format(str(self.img_fm.shape)))

            # Update Mask Size after feature map is captured 
            self.mask = self.mask.expand_as(self.img_fm)
            print('StyleLossPass1 mask expand to shape {}'.format(str(self.mask.shape)))

        elif self.mode == 'loss':
            # input in this case is the naive stiched image's fm 
            self.img_gram = self.gram(torch.mul(input, self.mask))
            self.img_gram = self.img_gram / torch.sum(self.mask)
            self.loss = self.critertain(self.img_gram, self.style_matched_gram) * self.weight

        # If None, do nothing 
        return input

    def match_fm(self, style_fm, img_fm):
        '''
        Input : 
            style_fm, img_fm : 1 * C * H * W

        Process:
            Instead of only compute the match for pixel inside the mask, here we compute the 
                match for the whole feature map and use mask to filter out the matched pixel we don't need 
            Patch Match is done in convolution fashion where content image's patch is used as kernal and applied 
                to the styled image, a score map will be computed (w.r.t each content fm patch) and we construct the 
                matched style feature map using the score map we get for each content fm patch 
        
        Output:
            style_fm_masked : 1 * C * H * W

        '''
        since = time.time()
        # TODO set a time count on how long it takes to compute the match 
        print('===>StyleLossPass1 Start to match feature map')

        # Create a copy of style image fm (to matain size)
        style_fm_masked = style_fm.clone()  # create a copy of the same size
        n_patch_h = math.floor(style_fm_masked.shape[2] / 3)  # use math package to avoid potential python2 issue
        n_patch_w = math.floor(style_fm_masked.shape[3] / 3)

        # Deyang and Ziyan ask why here stride is 3 instead of 1? the original code use 1.
        # Original Code location: neural_gram line 134, stride = 1
        # TODO sx change the stride to 1 and support new feature map 
        stride = self.match_patch_size
        patch_size = self.match_patch_size

        for i in range(n_patch_h):
            for j in range(n_patch_w):
                # Each kernal represent a patch in content fm 
                kernal = img_fm[:, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]

                # Compute score map for each content fm patch 
                score_map = F.conv2d(style_fm, kernal, stride=stride)  # 1 * 1 * n_patch_h * n_patch_w
                score_map = score_map[0, 0, :, :]  # n_patch_h * n_patch_w
                idx = torch.argmax(score_map).item()
                matched_style_idx = (int(idx / score_map.shape[1]), int(idx % score_map.shape[1]))

                # Find matched patch 
                matched_style_patch = style_fm[:, :,
                                      matched_style_idx[0] * patch_size: (matched_style_idx[0] + 1) * patch_size, \
                                      matched_style_idx[1] * patch_size: (matched_style_idx[
                                                                              1] + 1) * patch_size]  # matched have shape 1 * C * 3 * 3

                style_fm_masked[:, :, i * patch_size:(i + 1) * patch_size,
                j * patch_size:(j + 1) * patch_size] = matched_style_patch

        assert (style_fm_masked.shape == img_fm.shape)

        time_elapsed = time.time() - since
        print('===>StyleLossPass1 Finish matching feature map with time{:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                                                               time_elapsed % 60))

        return style_fm_masked


class StyleLossPass2(StyleLossPass1):
    '''
    child class of StyleLossPass1 that's capable of compute nearest neighbor like pass 1 
    '''

    def __init__(self, style_weight, layer_mask, match_patch_size):
        super(StyleLossPass2, self).__init__(style_weight, layer_mask, match_patch_size)
        # TODO

    def forward(self):
        return None

    def match_fm(self, style_fm, img_fm, ref_layer_idx):
        # after conv, normalize, loc: cuda_utils line 1260
        

        return None
