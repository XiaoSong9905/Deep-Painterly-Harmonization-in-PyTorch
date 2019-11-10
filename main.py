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


def get_args():
    parser = argparse.ArgumentParser()
    # Input Output
    parser.add_argument("-style_image", help="./path/file to style image", default='data/0_target.jpg')
    parser.add_argument("-native_image", help="./path/file to simple stich image", default='data/0_naive.jpg')
    parser.add_argument("-tight_mask", help="./path/file to tight mask", default='data/0_c_mask.jpg')
    parser.add_argument("-dilated_mask", help="./path/file to dilated(loss) mask", default='data/0_c_mask_dilated.jpg')
    parser.add_argument("-p1_output_img", help="./path/file for output image", default='output/0_pass1_out.jpg')
    parser.add_argument("-p2_output_img", help="./path/file for output image", default='output/0_pass2_out.jpg')
    parser.add_argument("-output_img_size", help="Max side(H/W) for output image, power of 2 is recommended", type=int,
                        default=512)

    # Training Parameter
    parser.add_argument("-optim", choices=['lbfgs', 'adam'], default='adam')
    parser.add_argument("-lr", type=float, default=1e-1)
    parser.add_argument("-p1_n_iters", type=int, default=1000)
    parser.add_argument("-p2_n_iters", type=int, default=1000)
    parser.add_argument("-print_interval", type=int, default=100)
    parser.add_argument("-save_img_interval", type=int, default=100)
    parser.add_argument("-log_interval", type=int, default=100)
    parser.add_argument("-gpu", help="Zero-indexed ID of the GPU to use; for CPU mode set -gpu = cpu", default='cpu')

    # Model Parameter
    parser.add_argument("-p1_content_layers", help="layers for content", default='relu4_2')
    parser.add_argument("-p1_style_layers", help="layers for style", default='relu1_1,relu2_1,relu3_1,relu4_1,relu5_1')
    parser.add_argument("-p2_content_layers", help="single layer for content", default='relu4_2')
    parser.add_argument("-p2_style_layers", help="single layer for style", default='relu4_2')
    parser.add_argument("-content_weight", type=float, default=0.5)
    parser.add_argument("-style_weight", type=float,
                        default=2)  # can setup content loss, style loss seprately for pass1 pass2
    parser.add_argument("-tv_weight", type=float, default=1e-3)
    parser.add_argument("-init", choices=['random', 'image'], default='image')
    parser.add_argument("-model_file", help="path/file to saved model file, if not will auto download", default=None)
    parser.add_argument("-model", choices=['vgg16', 'vgg19', 'resnet18', 'resnet34'], default='vgg16')
    parser.add_argument("-match_patch_size", type=int, default=3)

    # Other option
    parser.add_argument("-debug_mode", type=bool, default=True)
    parser.add_argument("-normalize_gradient", type=bool,
                        default=False)  # TODO normalize gradient is currently not supported
    parser.add_argument("-mask_gradient_bp", type=bool,
                        default=False)  # TODO implementation detail, wether mask gradient during backprop

    cfg = parser.parse_args()
    return cfg


def preprocess(cfg):
    # Set up device and datatye
    dtype, device = setup(cfg)

    # Get input image and preprocess
    native_img = img_preprocess(cfg.native_image, cfg.output_img_size, dtype, device, cfg,
                                name='read_naive_img.png').type(dtype)
    style_img = img_preprocess(cfg.style_image, cfg.output_img_size, dtype, device, cfg,
                               name='read_style_img.png').type(dtype)
    tight_mask = mask_preprocess(cfg.tight_mask, cfg.output_img_size, dtype, device, cfg,
                                 name='read_tight_mask.png').type(dtype)
    loss_mask = mask_preprocess(cfg.dilated_mask, cfg.output_img_size, dtype, device, cfg,
                                name='read_loss_mask.png').type(dtype)

    display_masked_region(native_img, style_img, loss_mask)

    return native_img, style_img, tight_mask, loss_mask


def train(cfg, native_img, loss_mask, content_loss_list, style_loss_list, device):
    def periodic_print(i_iter, c_loss, s_loss, total_loss):
        if i_iter % cfg.print_interval == 0:
            print('Iteration {} ; Content Loss {}; Style Loss {}; Total Loss {}'.format(
                str(i_iter), str(c_loss.item()), str(s_loss.item()), str(total_loss.item())))

    def periodic_save(i_iter):
        flag = (i_iter % cfg.save_img_interval == 0) or (i_iter == cfg.p1_n_iters)
        if flag:
            print('===>save image at iteration {}'.format(str(i_iter)))
            output_filename, file_extension = os.path.splitext(cfg.p1_output_img)
            if i_iter == cfg.p1_n_iters:
                filename = output_filename + str(file_extension)
            else:
                filename = str(output_filename) + "_iter_{0:08d}".format(i_iter) + str(file_extension)
            img_deprocessed = img_deprocess(native_img)

            img_deprocessed.save(str(filename))

    # Build optimizer and run optimizer
    def closure():
        optimizer.zero_grad()
        _ = net(native_img)
        c_loss = 0
        s_loss = 0
        total_loss = 0

        for i in content_loss_list:
            c_loss += i.loss.to(device)
        for i in style_loss_list:
            s_loss += i.loss.to(device)

        total_loss = s_loss + c_loss
        total_loss.backward()

        native_img.grad = native_img.grad * loss_mask.expand_as(native_img)

        periodic_print(i_iter, c_loss, s_loss, total_loss)
        periodic_save(i_iter)

    optimizer = build_optimizer(cfg, native_img)
    # TODO
    # Addd schedular for this model, the output doesn't seem to converge
    i_iter = 0
    while i_iter <= cfg.p1_n_iters:
        optimizer.step(closure)
        i_iter += 1

    return native_img


def pass1(cfg, native_img, style_img, tight_mask, loss_mask):
    # Setup Network 
    content_layers = cfg.p1_content_layers.split(',')
    style_layers = cfg.p1_style_layers.split(',')
    content_loss_list = []
    style_loss_list = []
    tv_loss_list = []

    # Build backbone 
    cnn, layer_list = build_backbone(cfg)
    cnn = copy.deepcopy(cnn)

    # Build net with loss model 
    net = nn.Sequential()
    mask = loss_mask

    if cfg.debug_mode:
        print('===>build net with loss module')

    # TODO in order to save computation time, only use the cnn untile the last layer of style / content loss 

    for i, layer in enumerate(list(cnn)):
        if isinstance(layer, nn.Conv2d):
            net.add_module(str(len(net)), layer)
            sap = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            mask = sap(mask)
            # sap get a weighted mask, to see how this work, checkout the `understand mask` notebook 
        elif isinstance(layer, nn.ReLU):
            net.add_module(str(len(net)), layer)
        elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
            net.add_module(str(len(net)), layer)
            mask = F.interpolate(mask, scale_factor=(
                0.5, 0.5))  # resize 0.5 may not work, the network don't garentee a fix output

        # Add Loss layer 
        if layer_list[i] in content_layers:
            print('add content layer at net position {}'.format(str(len(net))))
            content_layer_loss = ContentLoss(cfg.content_weight, mask)
            net.add_module(str(len(net)), content_layer_loss)
            content_loss_list.append(content_layer_loss)

        if layer_list[i] in style_layers:
            print('add style layer at net position {}'.format(str(len(net))))
            # TODO style loss need more operation 
            # Match operation should be done at this stage, future forward backward only do the update 
            # See neural_gram.lua line 120 - 136 
            style_layer_loss = StyleLossPass1(cfg.style_weight, mask, cfg.match_patch_size)
            net.add_module(str(len(net)), style_layer_loss)
            style_loss_list.append(style_layer_loss)

    del cnn  # delet unused net to save memory

    net = net.to(device).eval()

    if cfg.debug_mode:
        print('===>build net')
        print('net is', net)
    # import pdb; pdb.set_trace()

    print('===>capture content fm')
    # Go pass the net to capture information we need 
    for i in content_loss_list:
        i.mode = 'capture'
    for i in style_loss_list:
        i.mode = 'capture_content'
    net(native_img)

    # import pdb; pdb.set_trace()
    print('===>capture style fm & compute match and style gram')
    for i in content_loss_list:
        i.mode = 'None'
    for i in style_loss_list:
        i.mode = 'capture_style'
    net(style_img)

    print('===>reset network to loss mode and freeze parameter')
    # reset the model to loss mode 
    for i in content_loss_list:
        i.mode = 'loss'

    for i in style_loss_list:
        i.mode = 'loss'

    for param in net.parameters():  # freeze new added loss layer
        param.requires_grad = False

    # Set image to be gradient updatable 
    native_img = nn.Parameter(native_img)
    native_img = native_img.to(device)
    assert (native_img.requires_grad == True)

    native_img = train(cfg, native_img, loss_mask, content_loss_list, style_loss_list, device)
    print('===>deep painterly harmonization pass 1 is done')
    return native_img


def pass2():
    # Set up device and datatye 
    # TODO
    return None


def main():
    cfg = get_args()
    native_img, style_img, tight_mask, loss_mask = preprocess(cfg)
    pass1(cfg, native_img, style_img, tight_mask, loss_mask)
    # pass2
    return


if __name__ == '__main__':
    main()
