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


def get_patch(padding_feature, pos_h, pos_w, patch_size=3):
    '''
    return patch from feature at (pos_h, pos_w)

    :param padding_feature: 1 * C * (H + pad) * (W + pad)
    :param pos_h:
    :param pos_w:
    :param patch_size:
    :return: patch: 1 * C * patch_size * patch_size
    '''
    pos_h = int(pos_h)
    pos_w = int(pos_w)
    return padding_feature[:, :, pos_h: pos_h + patch_size, pos_w: pos_w + patch_size].clone()


def build_backbone(cfg):
    '''
    Notice : 
        1. User must specify a model weight 
        2. To use default model weight, run 'models/download_models.py' to download model 
    '''

    assert (cfg.model_file is not None)

    if cfg.model == 'vgg16':
        net, layer_list = models.vgg16(pretrained=False), vgg16_dict
    elif cfg.model == 'vgg19':
        net, layer_list = models.vgg19(pretrained=False), vgg19_dict
    else:
        print('Model Not support, use vgg16 / vgg19')
        exit(1)

    # When loading model, we asssume the model weight match the model architrcture 
    print('Build {} with weight {}'.format(cfg.model, cfg.model_file))
    user_state_dict = torch.load(cfg.model_file)
    net_state_dict = net.state_dict()
    user_state_dict = {k: v for k, v in user_state_dict.items() if k in net_state_dict}
    net_state_dict.update(user_state_dict)
    net.load_state_dict(net_state_dict)

    # Only use the Convolutional part of the model 
    net = net.features

    net = net.eval()

    for param in net.parameters():
        param.requires_grad = False

    assert (net.training == False)
    assert (len(net) == len(layer_list))

    return net, layer_list


class TVLoss(nn.Module):
    def __init__(self, weight):
        super(TVLoss, self).__init__()
        self.weight = weight

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
        # tmp_h = torch.sum((input[:, :, 1:, :] - input[:, :, :-1, :]) ** 2)
        # tmp_w = torch.sum((input[:, :, :, 1:] - input[:, :, :, :-1]) ** 2)
        # self.loss = self.weight * (tmp_h + tmp_w)
        # return input
        self.x_diff = input[:, :, 1:, :] - input[:, :, :-1, :]
        self.y_diff = input[:, :, :, 1:] - input[:, :, :, :-1]
        self.loss = self.weight * (torch.sum(torch.abs(self.x_diff)) + torch.sum(torch.abs(self.y_diff)))
        return input


class ContentLoss(nn.Module):
    def __init__(self, weight, mask):
        super(ContentLoss, self).__init__()
        self.weight = weight
        self.criterian = nn.MSELoss()
        self.mask = mask.clone()  # a weighted mask, not binary mask. To see why check `understand mask` notebook
        self.mode = 'None'
        self.loss = 0

    def forward(self, input):
        '''
        Process:
            1. before update image(Step 1), `self.mode = 'capture'` will be set to capture content image feature map and save 
            2. during update image(Step 2), `self.mode = 'loss'` will be set to compute loss and pass `input` to the next module
        '''
        # Step 1 : capture content image feature map 
        if self.mode == 'capture':
            # Capture Feature Map
            self.content_fm = input.detach()
            print('ContentLoss content feature map with shape {} captured'.format(str(self.content_fm.shape)))

            # Update Mask Size after feature map is captured 
            self.mask = self.mask.expand_as(self.content_fm)  # 1 * 1 * H * W -> 1 * C * H * W

        # Step 2 : compute loss 
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
    Take Reference from https://github.com/ProGamerGov/neural-style-pt 
    To understand how gram matrix work, checkout `understand Gram Matrix` notebook 
    '''

    def __init__(self):
        super(GramMatrix, self).__init__()

    def forward(self, input):
        '''
        Input : 
            input: B * C * H * W, represent feature map 
        Output : 
            output : B * (C * C), represent gram matrix 
        '''
        B, C, H, W = input.shape
        output = torch.zeros((B, C, C))

        for i in range(B):
            fm_flat = input[i].view(C, H * W)
            output[i] = torch.mm(fm_flat, fm_flat.t())

        return output

        # B, C, H, W = input.size()
        # x_flat = input.view(C, H * W)
        # return torch.mm(x_flat, x_flat.t())


class HistogramLoss(nn.Module):
    def __init__(self, weight, mask):
        super().__init__()
        self.R = None
        self.S = None
        self.weight = weight
        self.nbins = 256
        self.mode = 'None'
        self.loss = 0
        self.mask = mask

    # TODO: consider merge into forward
    # def find_match(self, input, idx):
    #     n1, c1, h1, w1 = self.content_L.shape
    #     n2, c2, h2, w2 = input.shape
    #     self.content_L.resize_(h1 * w1 * h2 * w2)
    #     input.resize_(h2 * w2 * h2 * w2)
    #     conv = torch.tensor((), dtype=torch.float32)
    #     conv = conv.new_zeros((h1 * w1, h2 * w2))
    #     conv.resize_(h1 * w1 * h2 * w2)
    #     assert c1 == c2, 'content:c{} is not equal to style:c{}'.format(c1, c2)
    #
    #     size1 = h1 * w1
    #     size2 = h2 * w2
    #     N = h1 * w1 * h2 * w2
    #     print('N is', N)
    #
    #     for i in range(0, N):
    #         i1 = i / size2
    #         i2 = i % size2
    #         x1 = i1 % w1
    #         y1 = i1 / w1
    #         x2 = i2 % w2
    #         y2 = i2 / w2
    #         kernal_radius = int((self.nbins - 1) / 2)
    #
    #         conv_result = 0
    #         norm1 = 0
    #         norm2 = 0
    #         dy = -kernal_radius
    #         dx = -kernal_radius
    #         while dy <= kernal_radius:
    #             while dx <= kernal_radius:
    #                 xx1 = x1 + dx
    #                 yy1 = y1 + dy
    #                 xx2 = x2 + dx
    #                 yy2 = y2 + dy
    #                 if 0 <= xx1 < w1 and 0 <= yy1 < h1 and 0 <= xx2 < w2 and 0 <= yy2 < h2:
    #                     _i1 = yy1 * w1 + xx1
    #                     _i2 = yy2 * w2 + xx2
    #                     for c in range(0, c1):
    #                         term1 = self.content_L[int(c * size1 + _i1)]
    #                         term2 = input[int(c * size2 + _i2)]
    #                         conv_result += term1 * term2
    #                         norm1 += term1 * term1
    #                         norm2 += term2 * term2
    #                 dx += self.stride
    #             dy += self.stride
    #         norm1 = math.sqrt(norm1)
    #         norm2 = math.sqrt(norm2)
    #         conv[i] = conv_result / (norm1 * norm2 + 1e-9)
    #
    #     match = torch.tensor((), dtype=torch.float32)
    #     match = match.new_zeros(self.content_L.size())
    #
    #     correspondence = torch.tensor((), dtype=torch.int16)
    #     correspondence.new_zeros((h1, w1, 2))
    #     correspondence.resize_(h1 * w1 * 2)
    #
    #     for id1 in range(0, size1):
    #         conv_max = -1e20
    #         for y2 in range(0, h2):
    #             for x2 in range(0, w2):
    #                 id2 = y2 * w2 + x2
    #                 id = id1 * size2 + id2
    #                 conv_result = conv[id1]
    #
    #                 if conv_result > conv_max:
    #                     conv_max = conv_result
    #                     correspondence[id1 * 2 + 0] = x2
    #                     correspondence[id1 * 2 + 1] = y2
    #
    #                     for c in range(0, c1):
    #                         match[c * size1 + id1] = input[c * size2 + id2]
    #
    #     match.resize_((n1, c1, h1, w1))
    #     self.masks[idx] = match
    #     return match, correspondence

    def find_nearest_above(self, my_array, target):
        diff = my_array - target
        mask = np.ma.less_equal(diff, -1)
        # We need to mask the negative differences
        # since we are looking for values above
        if np.all(mask):
            c = np.abs(diff).argmin()
            return c  # returns min index of the nearest if target is greater than any value
        masked_diff = np.ma.masked_array(diff, mask)
        return masked_diff.argmin()

    def hist_match(self, A, B):
        original = A.cpu().detach().numpy()
        specified = B.cpu().detach().numpy()

        oldshape = original.shape
        original = original.ravel()
        specified = specified.ravel()

        # get the set of unique pixel values and their corresponding indices and counts
        s_values, bin_idx, s_counts = np.unique(original, return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(specified, return_counts=True)

        # Calculate s_k for original image
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]

        # Calculate s_k for specified image
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # Round the values
        sour = np.around(s_quantiles * 255)
        temp = np.around(t_quantiles * 255)

        # Map the rounded values
        b = []
        for data in sour[:]:
            b.append(self.find_nearest_above(temp, data))
        b = np.array(b, dtype='uint8')

        return b[bin_idx].reshape(oldshape)

    def forward(self, input):
        if self.mode == 'capture_style':
            self.S = input.clone()
            print('His Loss Capture Style Image Feature Map')

        elif self.mode == 'capture_inter':
            # TODO: calulate histmatch(content, input), then calculate R
            R = self.hist_match(input, self.S)
            import pdb; pdb.set_trace()
            
            self.R = torch.from_numpy(R).to(input.dtype, input.device)
            print('His Loss Capture Inter Image Feature Map & Compute Match')

        elif self.mode == 'loss':
            self.loss = self.weight * torch.sum((input - self.R) ** 2)
            self.loss = self.loss / input.nelement()

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


class StyleLossPass1(nn.Module):
    def __init__(self, weight, mask, match_patch_size, stride=1, device='cpu', verbose=False):
        super().__init__()
        self.weight = weight
        self.critertain = nn.MSELoss()
        self.gram = GramMatrix()
        self.mask = mask.clone()
        self.mode = 'None'
        self.patch_size = match_patch_size  # patch size for matching between feature map, in the original paper 3 is used
        self.stride = stride
        self.device = device
        self.verbose = verbose

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
        # Step 1 : Capture Content Feature Map  
        if self.mode == 'capture_content':
            self.content_fm = input.detach()
            if self.verbose:
                print('StyleLossPass1 content feature map with shape {} captured'.format(str(self.content_fm.shape)))

            # Update Mask Size after feature map is captured 
            self.mask = self.mask.expand_as(self.content_fm)

        # Step 2 : Capture Style Feature Map & Compute Match & Compute Gram 
        elif self.mode == 'capture_style':  #
            style_fm = input.detach()
            assert (style_fm.shape == self.content_fm.shape)
            if self.verbose:
                print('StyleLossPass1 style feature map with shape {} captured'.format(str(style_fm.shape)))

            # Compute Match 
            if self.weight > 0:
                correspond_fm, correspond_idx = self.match_fm(self.content_fm, style_fm)
            else:  # if weight 0, disable compute matching
                correspond_fm = style_fm
            if self.verbose:
                print('StyleLossPass1 compute match relation')

            # Compute Gram Matrix 
            self.G = self.gram(torch.mul(correspond_fm, self.mask)) / torch.sum(self.mask)
            # self.G = self.gram(correspond_fm) / correspond_fm.nelement()
            self.target = self.G.detach()
            if self.verbose:
                print('StyleLossPass1 compute style gram matrix')

            del self.content_fm
            del style_fm

        # Step 3 : during updateing image 
        elif self.mode == 'loss':
            # self.G = self.gram(input)
            # self.G = self.G / input.nelement()
            # self.loss = self.critertain(self.G, self.target) * self.weight
            self.G = self.gram(torch.mul(input, self.mask))
            self.G = self.G / torch.sum(self.mask)
            self.loss = self.critertain(self.G, self.target) * self.weight

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

    def match_fm(self, content_fm, style_fm):
        '''
        Input : 
            style_fm : 1 * C * H * W 
            content_fm : 1 * C * H * W
        Process:
            Instead of only compute the match for pixel inside the mask, here we compute the 
                match for the whole feature map and use mask to filter out the matched pixel we don't need 
            Patch Match is done in convolution fashion where content image's fm patch is used as kernal and applied 
                to the styled image fm, a score map will be computed (w.r.t each content fm location) and we construct the 
                matched style feature map using the score map we get for each content fm patch 
            After compute score map using covolutiopn, a normalization process is used to avoid overfloat and local maximal issue
        
        Score Map Normalize Process:
            patch1 : 3 * 3 
            patch2 : 3 * 3 
            score = sum(patch1 * patch2)
            norm1 = sum(patch1**2)**0.5 
            norm2 = sum(patch2**2)**0.5 
            score = score / (norm1 * norm2 + 1e-9)

            When used at fm level, 
                # norm_style = style_fm_pad**2 
                # norm_style = conv(norm_style, kernal=3*3(1))
                # norm_style = norm_style ** 0.5 
                # For each location i in content_fm_pad 
                #     kernal = patch = 3 * 3 patch around location i 
                #     score_map = conv(style_fm_pad, kernal, stride=1)
                #     norm_kernal = sum(kernal**2)**0.5 
                #     score_map = score_map / (norm_style * norm_kernal + 1e-9 )
                #     correspondence_idx = argmax(score_map)  -> (x, y)
                #     correspondence = style_fm(idx)

        Notice : 
            to index into 4d Tensor [b, c, y, x] is used
        
        Output:
            correspond_fm  1 * C * H * W
            correspond_idx  2 * H * W first channel represent x index, second channel represent y index       
        '''

        # Padding Style FM & Content FM 
        stride = self.stride
        patch_size = self.patch_size
        padding = (patch_size - 1) // 2
        c1, h1, w1 = content_fm.shape[1], content_fm.shape[2], content_fm.shape[3]
        c2, h2, w2 = style_fm.shape[1], style_fm.shape[2], style_fm.shape[3]
        c, h, w = c1, h1, w1

        # It's not nessary for two feature map to share the same spatial dimention
        # but in this project we enforce that for better quantititive result 
        assert (content_fm.shape == style_fm.shape)

        n_patch_h = math.floor(h / stride)  # use math package to avoid potential python2 issue
        n_patch_w = math.floor(w / stride)

        style_fm_pad = F.pad(style_fm, (padding,) * 4, mode='reflect')  # 1 * C * (H + 2 * padding) * (W + 2 * padding)
        content_fm_pad = F.pad(content_fm, (padding,) * 4,
                               mode='reflect')  # 1 * C * (H + 2 * padding) * (W + 2 * padding)

        correspond_fm = style_fm.clone()  # 1 * C * H * W
        correspond_idx = torch.zeros((2, h, w))  # 2 * H * W where first layer is x, second layer is y

        # Compute Style FM Norm 
        style_fm_norm = style_fm_pad ** 2
        kernal = torch.ones((1, c, patch_size, patch_size)).to(self.device)
        style_fm_norm = F.conv2d(style_fm_norm, kernal, stride=stride, padding=0)
        style_fm_norm = style_fm_norm ** 0.5  # 1 * C * H * W

        # Compute Score Map for each location on content_fm
        for i in range(n_patch_h):  # H (y)
            for j in range(n_patch_w):  # W (x)

                # Each kernal represent a patch in content fm 
                kernal = content_fm_pad[:, :, i:i + patch_size, j:j + patch_size].clone()
                kernal_norm = torch.sum(kernal ** 2) ** 0.5

                # Compute score map for each content fm patch kernal 
                score_map = F.conv2d(style_fm_pad, kernal, stride=stride, padding=0)  # 1 * C * H * W

                # Normalize score map 
                score_map = score_map / (kernal_norm * style_fm_norm + 1e-9)
                score_map = score_map[0, 0, :, :]  # 1 * 1 * H * W -> H * W 

                # Find Maximal idx of score map 
                idx = torch.argmax(score_map).item()
                matched_style_idx = (int(idx // score_map.shape[1]), int(idx % score_map.shape[1]))  # (y, x)

                # Corresponding FM 
                # Index into 4d Tensor : [b, c, y, x]
                correspond_fm[:, :, i, j] = style_fm[:, :, matched_style_idx[0], matched_style_idx[1]]
                correspond_idx[0, i, j] = matched_style_idx[0]
                correspond_idx[1, i, j] = matched_style_idx[1]

        assert (correspond_fm.shape == content_fm.shape)

        return correspond_fm, correspond_idx  # 1 * C * H * W, 2 * H * W (first channel is x)


class StyleLossPass2(StyleLossPass1):
    '''
    child class of StyleLossPass1 that's capable of compute nearest neighbor like pass 1 
    '''

    def __init__(self, weight, mask, match_patch_size, stride=1, device='cpu', verbose=False):
        super(StyleLossPass2, self).__init__(weight, mask, match_patch_size, stride, device, verbose)
        self.ref_corr = None
        # TODO

    def forward(self, input):
        # Step 2: Capture Style Feature Map & Compute Match & Compute Gram for ref layer
        if self.mode == 'capture_style_ref':
            style_fm = input.detach()
            if self.verbose:
                print('StyleLossPass2 style feature map ref layer with shape {} captured'.format(str(style_fm.shape)))

            # Compute Match
            self.ref_corr, self.style_fm_matched = self.match_fm_ref(style_fm, self.content_fm)
            if self.verbose:
                print('StyleLossPass2 compute match relation')
            # Compute Gram Matrix
            style_fm_matched_masked = torch.mul(self.style_fm_matched, self.mask)
            self.target_gram = self.gram(style_fm_matched_masked) / torch.sum(self.mask)
            if self.verbose:
                print('StyleLossPass2 compute style gram matrix')

        # Step 3: Capture Style Feature Map & Compute Match & Compute Gram for other layers
        elif self.mode == 'capture_style_others':
            if self.ref_corr is None:
                if self.verbose:
                    print('No ref_corr infor, do nothing.')
                return input
            style_fm = input.detach()
            _, _, curr_H, curr_W = input.shape
            _, self.style_fm_matched = self.upsample_corr(self.ref_corr, curr_H, curr_W, style_fm)
            style_fm_matched_masked = torch.mul(self.style_fm_matched, self.mask)
            self.target_gram = self.gram(style_fm_matched_masked) / torch.sum(self.mask)
            if self.verbose:
                print('StyleLossPass2 compute style gram matrix')

        # Step 1 : Capture Content Feature Map
        elif self.mode == 'capture_content':
            self.content_fm = input.detach()
            if self.verbose:
                print('StyleLossPass2 content feature map with shape {} captured'.format(str(self.content_fm.shape)))

            # Update Mask Size after feature map is captured
            self.mask = self.mask.expand_as(self.content_fm)

        # Step 4 : during updateing image
        elif self.mode == 'loss':
            self.img_gram = self.gram(torch.mul(input, self.mask))
            self.img_gram = self.img_gram / torch.sum(self.mask)
            self.loss = self.critertain(self.img_gram, self.target_gram) * self.weight

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

        ref_h = style_fm.shape[2]  # height of the reference layer
        ref_w = style_fm.shape[3]  # width of the reference layer
        n_patch_h = math.floor(ref_h / stride)  # the number of patches along height
        n_patch_w = math.floor(ref_w / stride)  # the number of patches along width
        padding_style_fm = F.pad(style_fm, [padding, padding, padding, padding])  # TODO delete mode='reflect'
        # padding_img_fm = F.pad(img_fm, [padding, padding, padding, padding]) # TODO delete mode='reflect'

        # nearest neighbor index for ref_layer: H_ref * W_ref, same as P_out in paper
        ref_corr = np.zeros((ref_h, ref_w))  # Output

        # Step 1: Find matches for the reference layer.
        _, corr_tmp = super().match_fm(img_fm, style_fm)

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

                        patch_idx = corr_tmp[:, i + di, j + dj]  # index of neighbor patch in style feature map
                        patch_pos = (patch_idx[0] - di, patch_idx[1] - dj)

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

                            patch_idx = corr_tmp[:, i + di, j + dj]
                            patch_pos = (patch_idx[0], patch_idx[1])

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
                # TODO MAYBE WRONG 
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
                ref_idx = [(i + 0.4999) // h_ratio, (j + 0.4999) // w_ratio]
                ref_idx[0] = int(max(min(ref_idx[0], ref_h - 1), 0))
                ref_idx[1] = int(max(min(ref_idx[1], ref_w - 1), 0))

                ref_mapping_idx = ref_corr[ref_idx[0], ref_idx[1]]
                ref_mapping_idx = (ref_mapping_idx // ref_w, ref_mapping_idx % ref_w)

                curr_mapping_idx = (int(i + (ref_mapping_idx[0] - ref_idx[0]) * h_ratio + 0.4999),
                                    int(j + (ref_mapping_idx[1] - ref_idx[1]) * w_ratio + 0.4999))
                curr_corr[i, j] = curr_mapping_idx[0] * curr_w + curr_mapping_idx[1]

                #  1 * C * 3 * 3
                style_fm_matched[:, :, i, j] = style_fm[:, :, curr_mapping_idx[0], curr_mapping_idx[1]]

        return curr_corr, style_fm_matched
