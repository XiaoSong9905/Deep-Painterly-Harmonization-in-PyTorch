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

from model import * 
from utils import * 

if not os.path.exists('output'):
    os.makedirs('output')

parser = argparse.ArgumentParser()
# Input Output 
parser.add_argument("-style_image", help="./path/file to style image", default='data/0_target.jpg')
parser.add_argument("-native_image", help="./path/file to simple stich image", default='data/0_naive.jpg')
parser.add_argument("-tight_mask", help="./path/file to tight mask", default='data/0_c_mask.jpg')
parser.add_argument("-dilated_mask", help="./path/file to dilated(loss) mask", default='data/0_c_mask_dilated.jpg')
parser.add_argument("-output_img", help="./path/file for output image", default='output/0_output.jpg')
parser.add_argument("-output_img_size", help="Max side(H/W) for output image", type=int, default=500)

# Training Parameter 
parser.add_argument("-optim", choices=['lbfgs', 'adam'], default='lbfgs')
parser.add_argument("-lr", type=float, default=1e-2)
parser.add_argument("-n_iters", type=int, default=1000)
parser.add_argument("-print_interval", type=int, default=100)
parser.add_argument("-save_img_interval", type=int, default=100)
parser.add_argument("-log_interval", type=int, default=100)
parser.add_argument("-gpu", help="Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1", default=-1)

# Model Parameter 
parser.add_argument("-content_layers", help="layers for content", default='relu4_2')
parser.add_argument("-style_layers", help="layers for style", default='relu1_1,relu2_1,relu3_1,relu4_1,relu5_1')
parser.add_argument("-content_weight", type=float, default=5)
parser.add_argument("-style_weight", type=float, default=100)
parser.add_argument("-tv_weight", type=float, default=1e-3)
parser.add_argument("-init", choices=['random', 'image'], default='image')
parser.add_argument("-model_file", help="path/file to saved model file, if not will auto download", default=None)
parser.add_argument("-model", choices=['vgg16', 'vgg19', 'resnet18', 'resnet34'], default='vgg16')

# Other option 
parser.add_argument("-debug_mode", type=bool, default=True)

cfg = parser.parse_args()
    
def main():
    # Set up device and datatye 
    dtype, device = setup(cfg)

    # Get input image and preprocess 
    native_img = img_preprocess(cfg.native_image, cfg.output_img_size, dtype, device, cfg, name='read_naive_img.png')
    style_img = img_preprocess(cfg.style_image, cfg.output_img_size, dtype, device, cfg,name='read_style_img.png')
    tight_mask = mask_preprocess(cfg.tight_mask, cfg.output_img_size, dtype, device, cfg,name='read_tight_mask.png')
    loss_mask = mask_preprocess(cfg.dilated_mask, cfg.output_img_size, dtype, device, cfg,name='read_loss_mask.png')
    style_img = nn.Parameter(style_img)
    assert(style_img.requires_grad==True)

    # Setup Network 
    content_layers = cfg.content_layers.split(',')
    style_layers = cfg.style_layers.split(',')
    content_loss = []
    style_loss = []
    tv_loss = []

    # Build backbone 
    cnn, layer_list = build_backbone(cfg, device)
    cnn = copy.deepcopy(cnn)

    # Build net with loss model 
    net = nn.Sequential()
    mask = loss_mask

    for i, layer in enumerate(list(cnn)):
        # Add Backbone Layer 
        # If cnn backbone have other module, add other module in this stage         
        if isinstance(layer, nn.Conv2d):
            net.add_module(str(len(net)), layer)
            sap = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            #print('mask before sap', mask)
            #torchvision.utils.save_image(mask, '{}.png'.format('sap_mask_'+str(i)+'_before' ))
            #save_img_plt(mask, 'sap_mask_'+str(i)+'_before.png', gray=True)
            mask = sap(mask)
            #save_img_plt(mask, 'sap_mask_'+str(i)+'_after.png', gray=True)
            #torchvision.utils.save_image(mask, '{}.png'.format('sap_mask_'+str(i)+'_after' ))
            #print('mask after sap', mask)
        elif isinstance(layer, nn.ReLU):
            net.add_module(str(len(net)), layer)
        elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
            net.add_module(str(len(net)), layer)
            #resize = transforms.Resize((math.floor(mask.shape[1]/2), math.floor(mask.shape[0]/2)))
            #torchvision.utils.save_image(mask, '{}.png'.format('resize_mask_'+str(i)+'_before'))
            #save_img_plt(mask, 'resize_mask_'+str(i)+'_before.png', gray=True)
            #print('mask dim', mask.dim())
            #print(mask.shape)
            mask = F.interpolate(mask, scale_factor=(0.5, 0.5))
            #mask = F.interpolate(mask, size=(math.floor(mask.shape[1]/2), math.floor(mask.shape[0]/2)))
            #save_img_plt(mask, 'resize_mask_'+str(i)+'_after.png', gray=True)
            #mask = resize(mask)
            #torchvision.utils.save_image(mask, '{}.png'.format('resize_mask_'+str(i)+'_after'))

        # Add Loss layer 
        if layer_list[i] in content_layers:
            print('add content layer at {}'.format(str(len(net))))
            content_layer_loss = ContentLoss(cfg.content_weight, mask)
            net.add_module(str(len(net)), content_layer_loss)
            content_loss.append(content_layer_loss)
            
        if layer_list[i] in style_layers:
            print('add style layer at {}'.format(str(len(net))))
            # TODO style loss need more operation 
            style_layer_loss = StyleLossPass1()
            net.add_module(str(len(net)), style_layer_loss)
            style_loss.append(style_layer_loss)
    
    if cfg.debug_mode:
        print('===>build net')
        print('net is', net)



if __name__ == '__main__':
    main()