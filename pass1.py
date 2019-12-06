# EECS 442 @ UMich Final Project 
# No commercial Use Allowed 

import os
import torch
import torchvision
import torchsummary as summary 

import sys
sys.path.append(".")

from model import *
from utils import *

if not os.path.exists('output'):
    os.makedirs('output')

def train(cfg, device, net, content_loss_list, style_loss_list, tv_loss_list, histogram_loss_list, start_img, mask):
    print('\n===> Start Updating Image')
    start_time = time.time()

    net = net.to(device).eval()
    for param in net.parameters():
        param.requires_grad = False
    
    img = start_img.clone()
    img = nn.Parameter(start_img)

    content_loss_his = []
    style_loss_his = []
    tv_loss_his = []
    histogram_loss_his = []

    def periodic_print(i_iter, c_loss, s_loss, tv_loss, h_loss, total_loss):
        if i_iter % cfg.print_interval == 0:
            if cfg.tv_weight > 0:
                tv_loss = tv_loss.item()
            if cfg.histogram_weight > 0:
                h_loss = h_loss.item()
            print('Iteration {:06d}; Total Loss {:.06f}; Content Loss {:.06f}; Style Loss {:.06f}; \
TV Loss {:.06f}; Histogram Loss {:.06f}'.format(i_iter, total_loss, c_loss.item(), s_loss.item(), tv_loss, h_loss))
            '''
            for i, module in enumerate(content_loss_list):
                print("  Content " + str(i+1) + " loss: " + str(module.loss.item()))
            for i, module in enumerate(style_loss_list):
                print("  Style " + str(i+1) + " loss: " + str(module.loss.item()))     
            if cfg.histogram_weight > 0:
                for i, module in enumerate(histogram_loss_list):
                    print("  Histogram " + str(i+1) + " loss: " + str(module.loss.item()))
            if cfg.tv_weight > 0:
                for i, module in enumerate(tv_loss_list):
                    print("  Total Variance " + str(i+1) + " loss: " + str(module.loss.item()))
            '''

    def periodic_save_img(i_iter):
        flag = (i_iter % cfg.save_img_interval == 0) or (i_iter == cfg.n_iter)
        if flag:
            print('Iteration {:06d} Save Image'.format(i_iter))
            output_filename, file_extension = os.path.splitext(cfg.output_img)
            if i_iter == cfg.n_iter:
                filename = str(output_filename) + str(file_extension)
            else:
                filename = str(output_filename) + "_iter_{:06d}".format(i_iter) + str(file_extension)

            img_deprocessed = img_deprocess(img.clone())
            img_deprocessed.save(str(filename))
    
    def periodic_save_loss(i_iter, c_loss, s_loss, tv_loss, h_loss):
        if i_iter % 10 == 0:
            if cfg.tv_weight > 0:
                tv_loss = tv_loss.item()
            if cfg.histogram_weight > 0:
                h_loss = h_loss.item()
            content_loss_his.append(c_loss.item())
            style_loss_his.append(s_loss.item())
            tv_loss_his.append(tv_loss)
            histogram_loss_his.append(h_loss)

    # Build optimizer and run optimizer
    def closure():

        optimizer.zero_grad()
        _ = net(img)

        c_loss = 0
        s_loss = 0
        tv_loss = 0
        h_loss = 0
        total_loss = 0

        for i in content_loss_list:
            c_loss += i.loss.to(device)
        for i in style_loss_list:
            s_loss += i.loss.to(device)
        if cfg.tv_weight > 0:
            for i in tv_loss_list:
                tv_loss += i.loss.to(device)
        if cfg.histogram_weight > 0: # For pass1, this part is not used
            for i in histogram_loss_list:
                h_loss += i.loss.to(device)

        total_loss = s_loss + c_loss + tv_loss + h_loss# loss value is already weighted 
        total_loss.backward()

        # After computing gradient w.r.t img, only update gradient on the masked region of img 
        img.grad = torch.mul(img.grad, mask.expand_as(img))

        periodic_print(i_iter, c_loss, s_loss, tv_loss, h_loss, total_loss)
        periodic_save_img(i_iter)
        periodic_save_loss(i_iter, c_loss, s_loss, tv_loss, h_loss)

        return total_loss

    optimizer = build_optimizer(cfg, img)
    i_iter = 0
    while i_iter <= cfg.n_iter:
        optimizer.step(closure)
        i_iter += 1

    time_elapsed = time.time() - start_time
    print('@ Time Spend {:.04f} m {:.04f} s'.format(time_elapsed // 60, time_elapsed % 60))
    
    # Plot History 
    if cfg.tv_weight > 0:
        tv_loss_his = None
    if cfg.histogram_weight > 0:
        histogram_loss_his = None
    plt_plot_loss(content_loss_his, style_loss_his, tv_loss_his, histogram_loss_his)

    return img # 1 * 3 * H * W 


def build_net(cfg, device, mask, StyleLoss, ContentLoss, TVLoss, HistogramLoss):
    # Setup Network 
    content_layers = cfg.content_layers.split(',')
    style_layers = cfg.style_layers.split(',')
    histogram_layers = cfg.style_layers.split(',')
    content_loss_list = []
    style_loss_list = []
    tv_loss_list = []
    histogram_loss_list = [] # For pass1, will be empty list 

    # Build backbone 
    cnn, layer_list = build_backbone(cfg)
    cnn = copy.deepcopy(cnn)
    print('\n===> Build Backbone Network with {}'.format(cfg.model))
    print(cnn)

    # Build net with loss model 
    net = nn.Sequential()

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

            if cfg.mask_on == 'off':
                mask = torch.ones_like(mask) # Mask of all 1 is used, which means no mask is used

            content_layer_loss = ContentLoss(weight=cfg.content_weight, mask=mask)
            net.add_module(str(len(net)), content_layer_loss)
            content_loss_list.append(content_layer_loss)

        if layer_list[i] in style_layers:
            print('Add Style Loss at Position {}'.format(str(len(net))))

            if cfg.mask_on == 'off':
                mask = torch.ones_like(mask) # Mask of all 1 is used, which means no mask is used

            style_layer_loss = StyleLoss(weight=cfg.style_weight, mask=mask, match_patch_size=cfg.match_patch_size, stride=1, device=device)
            net.add_module(str(len(net)), style_layer_loss)
            style_loss_list.append(style_layer_loss)

        # For pass1, cfg.histogram_weight == 0, no histogram layer is added here
        if layer_list[i] in histogram_layers and cfg.histogram_weight > 0:
            print('Add Histogram Loss at Position {}'.format(str(len(net))))

            if cfg.mask_on == 'off':
                mask = torch.ones_like(mask) # Mask of all 1 is used, which means no mask is used

            histogram_layer_loss = HistogramLoss(weight=cfg.histogram_weight, mask=mask) # TODO add code for initializaiton 
            net.add_module(str(len(net)), histogram_layer_loss)
            histogram_loss_list.append(histogram_layer_loss)

    del cnn  # delet unused net to save memory

    net = net.to(device).eval()
    for param in net.parameters():
        param.requires_grad = False

    print(net)
    
    return content_loss_list, style_loss_list, tv_loss_list, histogram_loss_list, net

def capture_fm_pass1(content_loss_list, style_loss_list, tv_loss_list, content_img, style_img, net):

    print('\n===> Start Capture Content Image Feature Map')
    start_time = time.time()
    
    for i in content_loss_list: # For content loss 
        i.mode = 'capture'
    for i in style_loss_list: # For match relation 
        i.mode = 'capture_content'
    net(content_img)

    print('\n===> Start Capture Style Image Feature Map & Compute Matching Relation & Compute Target Gram Matrix')

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

    return None


def main():
    # Initial Config 
    cfg = get_args()

    # Setup Log 
    orig_stdout = init_log(cfg)

    # Initial Config 
    dtype, device = setup(cfg)
    content_img, style_img, inter_img, tight_mask, loss_mask = preprocess(cfg, dtype, device)

    # Build Network 
    content_loss_list, style_loss_list, tv_loss_list, histogram_loss_list, net = build_net(cfg, device, loss_mask, StyleLossPass1, ContentLoss, TVLoss, HistogramLoss)

    # Capture FM & Compute Match 
    capture_fm_pass1(content_loss_list, style_loss_list, tv_loss_list, content_img, style_img, net)

    # Training 
    inter_result = train(cfg, device, net, content_loss_list, style_loss_list, tv_loss_list, histogram_loss_list, start_img=content_img, mask=loss_mask)
    
    # End Log 
    end_log(orig_stdout)


if __name__ == '__main__':
    main()
