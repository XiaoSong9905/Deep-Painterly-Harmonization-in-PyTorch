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
from utils import get_patch

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

    # Build Model with user specific weight 
    if cfg.model_file is not None:
        user_pretrained_dict = torch.load(cfg.model_file)
        donwload_weight = False

    if cfg.model == 'vgg16':
        net, layer_list = models.vgg16(pretrained=donwload_weight).features, vgg16_dict
    elif cfg.model == 'vgg19':
        net, layer_list = models.vgg19(pretrained=donwload_weight).features, vgg19_dict
    else:
        return None

    # User Specific Weight 
    if user_pretrained_dict is not None:
        net_state_dict = net.state_dict()
        user_pretrained_dict = {k: v for k, v in user_pretrained_dict.items() if k in net_state_dict}
        net_state_dict.update(user_pretrained_dict)
        net.load_state_dict(net_state_dict)

    # Freeze Parameter, only update img not net parameter 
    # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/3 
    net = net.eval()
    for param in net.parameters():
        param.requires_grad = False

    print('\n===> Build Backbone Network with {}'.format(cfg.model))
    print(net)

    assert (net.training == False)
    assert (len(net) == len(layer_list))

    return net, layer_list


# Ziyan use class in order to match the format in this script. However, a function may better.
# Xiao : the reason to use a class is to able to put the TVLoss class into the network that can forward, backward by simply model(img), loss.backward()
class TVLoss(nn.Module):
    def __init__(self, tv_weight):
        super(TVLoss, self).__init__()
        self.weight = tv_weight

    def forward(self, input):
        '''
        Input : 
            input: pytorch variable represent img of shape (1,3,H,W)
        Return :
            input 
        Process:
            compute loss, save loss as part of TVLoss member element (so it's not released after forward pass)
        '''
        #####################
        # Formula: $L_{tv} = w_t \times \left(\sum_{c=1}^3\sum_{i=1}^{H-1}\sum_{j=1}^{W} (x_{i+1,j,c} - x_{i,j,c})^2 + \sum_{c=1}^3\sum_{i=1}^{H}\sum_{j=1}^{W - 1} (x_{i,j+1,c} - x_{i,j,c})^2\right)$
        #####################
        tmp_h = torch.sum((input[:, :, 1:, :] - input[:, :, :-1, :]) ** 2)
        tmp_w = torch.sum((input[:, :, :, 1:] - input[:, :, :, :-1]) ** 2)
        self.loss = self.weight * (tmp_h + tmp_w)
        return input


class ContentLoss(nn.Module):
    def __init__(self, content_weight, layer_mask):
        super(ContentLoss, self).__init__()
        self.weight = content_weight  # content loss weight
        self.criterian = nn.MSELoss()
        self.mask = layer_mask.clone()  # a weighted mask, not binary mask. To see why check `understand mask` notebook
        self.mode = 'None'

    def forward(self, input):
        '''
        Process:
            1. before update image, `self.mode = 'capture'` will be set to capture content image feature map and save 
            2. during update image, `self.mode = 'loss'` will be set to compute loss and pass `input` to the next module
        '''
        if self.mode == 'capture':
            # Capture Feature Map
            self.content_fm = input.detach()
            print('ContentLoss content feature map with shape {} captured'.format(str(self.content_fm.shape)))

            # Update Mask Size after feature map is captured 
            self.mask = self.mask.expand_as(self.content_fm)  # 1 * 1 * H * W -> 1 * C * H * W

        elif self.mode == 'loss':
            self.loss = self.criterian(input, self.content_fm) * self.weight

            def backward_variable_gradient_mask_hook_fn(grad):
                '''
                Functionality : 
                    Return Gradient only over masked region
                Notice : 
                    Variable hook is used in this case, Module hook is not supported for `complex moule` 
                '''
                return torch.mul(grad, self.mask)

            input.register_hook(backward_variable_gradient_mask_hook_fn)

        return input


class GramMatrix(nn.Module):
    '''
    Take Reference from https://github.com/jcjohnson/neural-style/blob/master/neural_style.lua 
    To understand how gram matrix work, checkout `understand Gram Matrix` notebook 
    '''

    def __init__(self):
        super(GramMatrix, self).__init__()

    def forward(self, input):
        '''
        Input : 
            input: 1 * C * H * W, represent feature map 
        Output : 
            output : 1 * (C * C), represent gram matrix 
        '''
        B, C, H, W = input.shape
        output = torch.zeros((B, C, C))

        for i in range(B):
            fm_flat = input[i].view(C, H * W)
            output[i] = torch.mm(fm_flat, fm_flat.t())

        return output


class StyleLossPass1(nn.Module):
    def __init__(self, style_weight, layer_mask, match_patch_size, stride=3):
        super().__init__()
        self.weight = style_weight
        self.critertain = nn.MSELoss()
        self.gram = GramMatrix()
        self.mask = layer_mask.clone()
        self.mode = 'None'
        self.loss = None
        self.patch_size = match_patch_size
        self.stride = stride

    def forward(self, input):
        '''
        Process:
            1. Before update image
                1.1 First set `self.mode = 'capture_content'` to capture content feature map & expanded mask to corresponding size 
                1.2 Then set `self.mode = 'capture_style'` to capture style feature map & compute match relation between 
                    style feature map and content feature map & compute style image gram matrix under masked region
            2. During update image, set `self.mode = 'loss'` to compute loss between content image gram matrix and stle image gram matrix 
               & return input
        '''
        # Step 2 : Capture Style Feature Map & Compute Match & Compute Gram 
        if self.mode == 'capture_style':  #
            style_fm = input.detach()
            print('StyleLossPass1 style feature map with shape {} captured'.format(str(style_fm.shape)))

            # Compute Match 
            style_fm_matched = self.match_fm(style_fm, self.content_fm)
            print('StyleLossPass1 compute match relation')

            # Compute Gram Matrix 
            style_fm_matched_masked = torch.mul(style_fm_matched, self.mask)
            self.style_matched_gram = self.gram(style_fm_matched_masked) / torch.sum(self.mask)
            print('StyleLossPass1 compute style gram matrix')

            del self.content_fm
            del style_fm

        # Step 1 : Capture Content Feature Map  
        elif self.mode == 'capture_content':
            self.content_fm = input.detach()
            print('StyleLossPass1 content feature map with shape {} captured'.format(str(self.content_fm.shape)))

            # Update Mask Size after feature map is captured 
            self.mask = self.mask.expand_as(self.content_fm)

        # Step 3 : during updateing image 
        elif self.mode == 'loss':
            self.img_gram = self.gram(torch.mul(input, self.mask))
            self.img_gram = self.img_gram / torch.sum(self.mask)
            self.loss = self.critertain(self.img_gram, self.style_matched_gram) * self.weight

        return input

    def match_fm(self, style_fm, content_fm):
        '''
        Input : 
            style_fm : 1 * C * H * W 
            content_fm : 1 * C * H * W
        Process:
            Instead of only compute the match for pixel inside the mask, here we compute the 
                match for the whole feature map and use mask to filter out the matched pixel we don't need 
            Patch Match is done in convolution fashion where content image's patch is used as kernal and applied 
                to the styled image, a score map will be computed (w.r.t each content fm patch) and we construct the 
                matched style feature map using the score map we get for each content fm patch 
        
        Output:
            style_fm_masked : 1 * C * H * W
                Step1 : compute corresponding feature map for whole content img feature map 
                Step2 : masked the matched feature map

        '''
        style_fm_matched = style_fm.clone()
        n_patch_h = math.floor(style_fm_matched.shape[2] / 3)  # use math package to avoid potential python2 issue
        n_patch_w = math.floor(style_fm_matched.shape[3] / 3)

        stride = self.stride
        patch_size = self.patch_size

        for i in range(n_patch_h):
            for j in range(n_patch_w):
                # Each kernal represent a patch in content fm 
                kernal = content_fm[:, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]

                # Compute score map for each content fm patch kernal 
                score_map = F.conv2d(style_fm, kernal, stride=stride)
                score_map = score_map[0, 0, :, :]  # 1 * 1 * n_patch_h * n_patch_w ->  1 * 1 * n_patch_h * n_patch_w

                # Find Maximal idx output score map 
                idx = torch.argmax(score_map).item()
                matched_style_idx = (int(idx / score_map.shape[1]), int(idx % score_map.shape[1]))

                # Find matched patch in style fm 
                matched_style_patch = style_fm[:, :,
                                      matched_style_idx[0] * patch_size: (matched_style_idx[0] + 1) * patch_size, \
                                      matched_style_idx[1] * patch_size: (matched_style_idx[1] + 1) * patch_size]
                #  1 * C * 3 * 3

                style_fm_matched[:, :,
                i * patch_size:(i + 1) * patch_size,
                j * patch_size:(j + 1) * patch_size] = matched_style_patch

        return style_fm_matched


class StyleLossPass2(StyleLossPass1):
    '''
    child class of StyleLossPass1 that's capable of compute nearest neighbor like pass 1 
    '''

    def __init__(self, style_weight, layer_mask, match_patch_size, stride=1):
        super(StyleLossPass2, self).__init__(style_weight, layer_mask, match_patch_size, stride)
        self.ref_corr = None
        # TODO

    def set_ref_infor(self, ref_corr):
        self.ref_corr = ref_corr

    def get_ref_infor(self):
        return self.ref_corr

    def forward(self, input):
        # Step 2: Capture Style Feature Map & Compute Match & Compute Gram for ref layer
        if self.mode == 'capture_style_ref':
            style_fm = input.detach()
            print('StyleLossPass2 style feature map ref layer with shape {} captured'.format(str(style_fm.shape)))

            # Compute Match
            self.ref_corr, self.style_fm_matched = self.match_fm_ref(style_fm, self.content_fm)
            print('StyleLossPass2 compute match relation')
            # Compute Gram Matrix
            style_fm_matched_masked = torch.mul(self.style_fm_matched, self.mask)
            self.style_matched_gram = self.gram(style_fm_matched_masked) / torch.sum(self.mask)
            print('StyleLossPass2 compute style gram matrix')

        # Step 3: Capture Style Feature Map & Compute Match & Compute Gram for other layers
        elif self.mode == 'capture_style_others':
            if self.ref_corr is None:
                print('No ref_corr infor, do nothing.')
                return input
            style_fm = input.detach()
            _, _, curr_H, curr_W = input.shape
            _, self.style_fm_matched = self.upsample_corr(self.ref_corr, curr_H, curr_W, style_fm)
            style_fm_matched_masked = torch.mul(self.style_fm_matched, self.mask)
            self.style_matched_gram = self.gram(style_fm_matched_masked) / torch.sum(self.mask)
            print('StyleLossPass2 compute style gram matrix')

        # Step 1 : Capture Content Feature Map
        elif self.mode == 'capture_content':
            self.content_fm = input.detach()
            print('StyleLossPass2 content feature map with shape {} captured'.format(str(self.content_fm.shape)))

            # Update Mask Size after feature map is captured
            self.mask = self.mask.expand_as(self.content_fm)

        # Step 4 : during updateing image
        elif self.mode == 'loss':
            self.img_gram = self.gram(torch.mul(input, self.mask))
            self.img_gram = self.img_gram / torch.sum(self.mask)
            self.loss = self.critertain(self.img_gram, self.style_matched_gram) * self.weight

        return input

    def match_fm_ref(self, style_fm, img_fm):
        '''
        Input :
            style_fm: 1 * C * H * W, reference layer style feature map
            img_fm: 1 * C * H * W, reference layer content feature map

        Output:
            ref_corr: H * W, reference layer mapping,
                      ref_corr[i, j] is the index of patch in style_fm (ref layer)
                      which matches the i_th row and j_th col patch in img_fm (ref layer)
            style_fm_masked : 1 * C * H * W
        '''

        stride = self.stride
        patch_size = self.patch_size
        padding = (patch_size - 1) // 2

        # Step 1: Find matches for the reference layer.
        ref_h = style_fm.shape[2]  # height of the reference layer
        ref_w = style_fm.shape[3]  # width of the reference layer
        n_patch_h = math.floor(ref_h / stride)  # the number of patches along height
        n_patch_w = math.floor(ref_w / stride)  # the number of patches along width
        padding_style_fm = F.pad(style_fm, [padding, padding, padding, padding])
        padding_img_fm = F.pad(img_fm, [padding, padding, padding, padding])

        corr_tmp = np.zeros((ref_h, ref_w))  # tmp variable, same as P in paper
        ref_corr = np.zeros((ref_h, ref_w))  # Output
        # nearest neighbor index for ref_layer: H_ref * W_ref, same as P_out in paper

        # for each patch
        for i in range(n_patch_h):
            for j in range(n_patch_w):
                # a patch in content fm
                patch = get_patch(padding_img_fm, i, j, patch_size)

                # Compute score map for each content fm patch
                score_map = F.conv2d(style_fm, patch, stride=stride, padding=padding)
                assert (score_map.shape[2:] == style_fm.shape[2:])
                score_map = score_map[0, 0, :, :]  # n_patch_h * n_patch_w
                corr_tmp[i, j] = torch.argmax(score_map).item()

        # Step 2: Enforce spatial consistency.
        for i in range(n_patch_h):
            for j in range(n_patch_w):
                # Initialize a set of candidate style patches.
                candidate_set = set()

                # For all adjacent patches, look up the corresponding style patch.
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        # skip if out of bounds
                        if i + di < 0 or i + di >= n_patch_h or j + dj < 0 or j + dj >= n_patch_w:
                            continue

                        patch_idx = corr_tmp[i + di, j + dj]  # index of neighbor patch in style feature map
                        patch_pos = (patch_idx // n_patch_w - di, patch_idx % n_patch_w - dj)

                        # skip if out of bounds
                        if patch_pos[0] < 0 \
                                or patch_pos[0] >= n_patch_h \
                                or patch_pos[1] < 0 \
                                or patch_pos[1] >= n_patch_w:
                            continue

                        candidate_set.add((patch_pos[0], patch_pos[1]))

                # Select the candidate the most similar to the style patches
                # associated to the neighbors of p.
                min_sum = np.inf
                for c_h, c_w in candidate_set:
                    style_fm_ref_c = get_patch(padding_style_fm, c_h, c_w,
                                               patch_size)  # get patch from style_fm at (c_h, c_w)
                    sum = 0

                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            # skip if out of bounds
                            if i + di < 0 or i + di >= n_patch_h or j + dj < 0 or j + dj >= n_patch_w:
                                continue

                            patch_idx = corr_tmp[i + di, j + dj]
                            patch_pos = (patch_idx // n_patch_w, patch_idx % n_patch_w)

                            # skip if out of bounds
                            if patch_pos[0] < 0 \
                                    or patch_pos[0] >= n_patch_h \
                                    or patch_pos[1] < 0 \
                                    or patch_pos[1] >= n_patch_w:
                                continue

                            # get patch from style_fm at (patch_pos[0], patch_pos[1])
                            style_fm_ref_p = get_patch(padding_style_fm, patch_pos[0], patch_pos[1], patch_size)
                            sum += F.conv2d(style_fm_ref_c, style_fm_ref_p).item()

                    if sum < min_sum:
                        min_sum = sum
                        ref_corr[i, j] = c_h * n_patch_w + c_w

        # Step 3: Create style_fm_matched based on ref_corr
        style_fm_matched = style_fm.clone()
        for i in range(n_patch_h):
            for j in range(n_patch_w):
                # Find matched index in style fm
                matched_style_idx = (int(ref_corr[i, j]) // ref_w, int(ref_corr[i, j]) % ref_w)
                #  1 * C * 3 * 3
                style_fm_matched[:, :, i, j] = style_fm[:, :, matched_style_idx[0], matched_style_idx[1]]

        return ref_corr.astype(np.int), style_fm_matched

    def upsample_corr(self, ref_corr, curr_h, curr_w, style_fm):
        '''
        Input :
            ref_corr: H * W, reference layer mapping,
                      ref_corr[i, j] is the index of patch in style_fm (ref layer)
                      which matches the i_th row and j_th col patch in img_fm (ref layer)
            curr_h: the height of current layer
            curr_w: the width of current layer
            style_fm : 1 * C * H * W

        Output:
            curr_corr: curr_h * curr_w, curr layer mapping,
                       curr_corr[i, j] is the index of patch in style_fm (current layer)
                       which matches the i_th row and j_th col patch in img_fm_curr (current layer)
            style_fm_masked : 1 * C * H * W
        '''

        # curr_corr = F.interpolate(torch.from_numpy(ref_corr), size=(curr_h, curr_w))
        # return curr_corr
        curr_corr = np.zeros((curr_h, curr_w))
        ref_h, ref_w = ref_corr.shape

        h_ratio = curr_h / ref_h
        w_ratio = curr_w / ref_w

        style_fm_matched = style_fm.clone()

        for i in range(curr_h):
            for j in range(curr_w):
                ref_idx = [(i + 0.5) // h_ratio, (j + 0.5) // w_ratio]
                ref_idx[0] = int(max(min(ref_idx[0], ref_w - 1), 0))
                ref_idx[1] = int(max(min(ref_idx[1], ref_h - 1), 0))

                ref_mapping_idx = ref_corr[ref_idx[0], ref_idx[1]]
                ref_mapping_idx = (ref_mapping_idx // ref_w, ref_mapping_idx % ref_w)

                curr_mapping_idx = (int(i + (ref_mapping_idx[0] - ref_idx[0]) * h_ratio + 0.5),
                                    int(j + (ref_mapping_idx[1] - ref_idx[1]) * w_ratio + 0.5))
                curr_corr[i, j] = curr_mapping_idx[0] * curr_w + curr_mapping_idx[1]

                #  1 * C * 3 * 3
                style_fm_matched[:, :, i, j] = style_fm[:, :, curr_mapping_idx[0], curr_mapping_idx[1]]

        return curr_corr, style_fm_matched
