# EECS 442 @ UMich Final Project 
# No commercial Use Allowed 

import os
import torch
import torchvision
from model import *
from utils import *
from pass1 import train, build_net

if not os.path.exists('output'):
    os.makedirs('output')

def capture_fm_pass2(content_loss_list, style_loss_list, tv_loss_list, histogram_loss_list, inter_img, content_img, style_img, net):
    '''
    Input:
        inter_img : output of pass1 algorithm 
        content_img : same content image as pass1 
        style_img : same style image as pass 1
    '''
    print('\n===> Capture Feature Map & Compute Style Loss Match & Compute Histogram Loss His')
    start_time = time.time()

    # Content Loss
    for i in content_loss_list:
        i.mode = 'capture'
    net(content_img)
    for i in content_loss_list: # Reset 
        i.mode = 'None' 

    # Style Loss 
    for i in style_loss_list:
        i.mode = 'capture_inter'
    net(inter_img)
    for i in style_loss_list: # Reset
        i.mode = 'None'

    for idx, i in enumerate(style_loss_list):  # TODO: change ref layer, and other layers
        if idx == len(style_loss_list) - 1:  # last layer
            i.mode = 'capture_style_ref'
        else:
            i.mode = 'capture_style_others'
    net(style_img)

    tmp_ref_corr = None
    for idx, i in reversed(list(enumerate(style_loss_list))):  # TODO: change ref layer, and other layers
        if not i.mode == 'capture_style_ref':
            i.ref_corr = tmp_ref_corr
        else:
            tmp_ref_corr = i.ref_corr
            i.mode = 'None'
    net(style_img)

    # Histogram Loss 
    histogram_loss_list[0].style_fm_matched = style_loss_list[0].style_fm_matched
    histogram_loss_list[1].style_fm_matched = style_loss_list[3].style_fm_matched
    for i in histogram_loss_list:
        i.compute_histogram() # compute histogram for style fm matched region inside mask 

    time_elapsed = time.time() - start_time
    print('@ Time Spend : {:.04f} m {:.04f} s'.format(time_elapsed // 60, time_elapsed % 60))

    # release memory 
    for i in style_loss_list:
        del i.style_fm_matched

    # reset the model to loss mode for update
    for i in content_loss_list:
        i.mode = 'loss'

    for i in style_loss_list:
        i.mode = 'loss'

    for i in histogram_loss_list:
        i.mode = 'loss'

    return None


def main():
    # Initial Config 
    cfg = get_args()

    # Setup Log 
    orig_stdout = init_log(cfg)

    # Initial Config 
    dtype, device = setup(cfg)
    content_img, style_img, inter_img, tight_mask, loss_mask = preprocess(cfg, dtype, device) # For Pass1, inter_img is the output of pass1

    # Build Network 
    content_loss_list, style_loss_list, tv_loss_list, histogram_loss_list, net = build_net(cfg, device, dtype, tight_mask, loss_mask, StyleLossPass2, ContentLoss, TVLoss, HistogramLoss)

    # Capture FM & Compute Match 
    capture_fm_pass2(content_loss_list, style_loss_list, tv_loss_list, histogram_loss_list, inter_img, content_img, style_img, net)

    # Training 
    final_img, content_loss_his, style_loss_his, tv_loss_his, histogram_loss_his = train(cfg, device, dtype, net, tight_mask, loss_mask, content_img, content_loss_list, style_loss_list, tv_loss_list, histogram_loss_list)
    
    # Plot History 
    plt_plot_loss(content_loss_his, style_loss_his, tv_loss_his, histogram_loss_his, name='pass2')

    # Crop output & save
    final_img = tight_mask_crop(cfg, final_img, style_img, tight_mask)

    # End Log 
    end_log(orig_stdout)


if __name__ == '__main__':
    main()
