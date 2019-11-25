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
import sys
from model import *
from utils import *

if not os.path.exists('output'):
    os.makedirs('output')

def get_args():
    parser = argparse.ArgumentParser()
    # Input Output
    parser.add_argument("-style_image", help="./path/file to style image", default='data/0_target.jpg')
    parser.add_argument("-native_image", help="./path/file to simple stich image", default='data/0_naive.jpg')
    parser.add_argument("-tight_mask", help="./path/file to tight mask", default='data/0_c_mask.jpg')
    parser.add_argument("-dilated_mask", help="./path/file to dilated(loss) mask", default='data/0_c_mask_dilated.jpg')
    parser.add_argument("-output_img", help="./path/file for output image", default='output/0_pass1_out.png')
    parser.add_argument("-output_img_size", help="Max side(H/W) for output image, power of 2 is recommended", type=int,
                        default=512)

    # Training Parameter
    parser.add_argument("-optim", choices=['lbfgs', 'adam'], default='adam')
    parser.add_argument("-lr", type=float, default=1e-1)
    parser.add_argument("-n_iter", type=int, default=1000)
    parser.add_argument("-print_interval", type=int, default=50)
    parser.add_argument("-save_img_interval", type=int, default=100)
    parser.add_argument("-log_interval", type=int, default=50)
    parser.add_argument("-gpu", help="Zero-indexed ID of the GPU to use; for CPU mode set -gpu = cpu", default='cpu')

    # Model Parameter
    parser.add_argument("-content_layers", help="layers for content", default='relu4_1')
    parser.add_argument("-style_layers", help="layers for style", default='relu3_1,relu4_1,relu5_1')
    parser.add_argument("-content_weight", type=float, default=5)
    parser.add_argument("-style_weight", type=float, default=100) 
    parser.add_argument("-tv_weight", type=float, default=1e-3)
    parser.add_argument("-init", choices=['random', 'image'], default='image')
    parser.add_argument("-model_file", help="path/file to saved model file, if not will auto download", default=None)
    parser.add_argument("-model", choices=['vgg16', 'vgg19'], default='vgg19')
    parser.add_argument("-match_patch_size", type=int, default=3)

    cfg = parser.parse_args()
    return cfg


def train(cfg, device, content_img, style_img, loss_mask, tight_mask, content_loss_list, style_loss_list, tv_loss_list, net):
    print('\n===> Start Updating Image')
    start_time = time.time()

    def periodic_print(i_iter, c_loss, s_loss, tv_loss, total_loss):
        if i_iter % cfg.print_interval == 0:
            print('Iteration {:06d} ; Content Loss {:.06f}; Style Loss {:.06f}; TV Loss {:.06f}; Total Loss {:.06f}'.format(
                i_iter, c_loss.item(), s_loss.item(), tv_loss.item(), total_loss.item() ) )

    def periodic_save(i_iter):
        flag = (i_iter % cfg.save_img_interval == 0) or (i_iter == cfg.n_iter)
        if flag:
            print('Iteration {:06d} Save Image'.format(i_iter))
            output_filename, file_extension = os.path.splitext(cfg.output_img)
            if i_iter == cfg.n_iter:
                filename = str(output_filename) + str(file_extension)
            else:
                filename = str(output_filename) + "_iter_{:06d}".format(i_iter) + str(file_extension)

            img_deprocessed = img_deprocess(content_img)
            img_deprocessed.save(str(filename))

    # Build optimizer and run optimizer
    def closure():

        optimizer.zero_grad()
        _ = net(content_img)

        c_loss = 0
        s_loss = 0
        tv_loss = 0
        total_loss = 0

        for i in content_loss_list:
            c_loss += i.loss.to(device)
        for i in style_loss_list:
            s_loss += i.loss.to(device)
        if cfg.tv_weight > 0:
            for i in tv_loss_list:
                tv_loss += i.loss.to(device)

        total_loss = s_loss + c_loss + tv_loss # loss value is already weighted 
        total_loss.backward()

        # After computing gradient w.r.t img, only update gradient on the masked region of img 
        #content_img.grad = content_img.grad * loss_mask.expand_as(content_img)

        periodic_print(i_iter, c_loss, s_loss, tv_loss, total_loss)
        periodic_save(i_iter)

    optimizer = build_optimizer(cfg, content_img)
    i_iter = 0
    while i_iter <= cfg.n_iter:
        optimizer.step(closure)
        i_iter += 1

    time_elapsed = time.time() - start_time
    print('@ Time Spend {:.04f} m {:.04f} s'.format(time_elapsed // 60, time_elapsed % 60))

    return content_img # 1 * 3 * H * W 


def build_net(cfg, device, content_img, style_img, tight_mask, loss_mask):
    # Setup Network 
    content_layers = cfg.content_layers.split(',')
    style_layers = cfg.style_layers.split(',')
    content_loss_list = []
    style_loss_list = []
    tv_loss_list = []

    # Build backbone 
    cnn, layer_list = build_backbone(cfg)
    cnn = copy.deepcopy(cnn)
    print('\n===> Build Backbone Network with {}'.format(cfg.model))
    print(cnn)

    # Build net with loss model 
    net = nn.Sequential()
    mask = loss_mask

    print('\n===> Build Network with Backbone & Loss Module')

    if cfg.tv_weight > 0:
        print('Add TVLoss at Position {}'.format(str(len(net))))
        tv_loss = TVLoss(cfg.tv_weight)
        net.add_module(str(len(net)), tv_loss)
        tv_loss_list.append(tv_loss)

    for i, layer in enumerate(list(cnn)):
        if isinstance(layer, nn.Conv2d):
            net.add_module(str(len(net)), layer)

            # sap get a weighted mask, to see how this work, checkout the `understand mask` notebook 
            sap = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            mask = sap(mask)

        elif isinstance(layer, nn.ReLU):
            net.add_module(str(len(net)), layer)

        elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
            net.add_module(str(len(net)), layer)

            # Scale the mask into the corresponding spatial size
            mask = F.interpolate(mask, scale_factor=(0.5, 0.5))  
            
        # Add Loss layer 
        if layer_list[i] in content_layers:
            print('Add Content Loss at Position {}'.format(str(len(net))))
            content_layer_loss = ContentLoss(cfg.content_weight, mask)
            net.add_module(str(len(net)), content_layer_loss)
            content_loss_list.append(content_layer_loss)

        if layer_list[i] in style_layers:
            print('Add Style Loss at Position {}'.format(str(len(net))))
            # TODO style loss need more operation 
            # Match operation should be done at this stage, future forward backward only do the update 
            # See neural_gram.lua line 120 - 136 
            style_layer_loss = StyleLossPass1(cfg.style_weight, mask, cfg.match_patch_size)
            net.add_module(str(len(net)), style_layer_loss)
            style_loss_list.append(style_layer_loss)

    del cnn  # delet unused net to save memory

    net = net.to(device).eval()
    for param in net.parameters():
        param.requires_grad = False

    print(net)

    print('\n===> Start Capture Content Image Feature Map')
    for i in content_loss_list: # For content loss 
        i.mode = 'capture'
    for i in style_loss_list: # For match relation 
        i.mode = 'capture_content'
    net(content_img)

    print('\n===> Start Capture Style Image Feature Map & Compute Matching Relation & Compute Target Gram Matrix')
    start_time = time.time()

    for i in content_loss_list:
        i.mode = 'None'
    for i in style_loss_list:
        i.mode = 'capture_style'
    net(style_img)

    time_elapsed = time.time() - start_time
    print('@ Time Spend : {:.04f} m {:.04f} s'.format(time_elapsed // 60, time_elapsed % 60))

    # reset the model to loss mode for update 
    for i in content_loss_list:
        i.mode = 'loss'

    for i in style_loss_list:
        i.mode = 'loss'

    # Set image to be gradient updatable 
    assert (content_img.requires_grad == True)

    return content_loss_list, style_loss_list, tv_loss_list, net


def main():
    # Setup Log 
    # orig_stdout = init_log()

    # Initial Config 
    cfg = get_args()
    dtype, device = setup(cfg)
    content_img, style_img, tight_mask, loss_mask = preprocess(cfg, dtype, device)

    # Build Network 
    content_loss_list, style_loss_list, tv_loss_list, net = build_net(cfg, device, content_img, style_img, tight_mask, loss_mask)

    # Training 
    inter_img = train(cfg, device, content_img, style_img, loss_mask, tight_mask, content_loss_list, style_loss_list, tv_loss_list, net)

    # End Log 
    # end_log(orig_stdout)


if __name__ == '__main__':
    main()
