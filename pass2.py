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

def capture_fm_pass2(content_loss_list, style_loss_list, tv_loss_list, histogram_loss_list, inter_img, content_img, style_img, net, verbose):
    if verbose:
        print('\n===> Start Capture Content Image Feature Map')
        start_time = time.time()
        print(len(content_loss_list), len(style_loss_list))

    # Content Loss
    for i in content_loss_list:
        i.mode = 'capture'
    net(content_img)

    for i in content_loss_list: # Reset 
        i.mode = 'None' 

    # Style Loss 
    for i in style_loss_list:
        i.mode = 'capture_content'
    net(content_img)

    for i in style_loss_list: # Reset
        i.mode = 'None'

    if verbose:
        print('\n===> Start Capture Style Image Feature Map & Compute Matching Relation & Compute Target Gram Matrix')
        print('total num of layers: ', len(style_loss_list), file=open('test.txt', 'w'))

    for idx, i in enumerate(style_loss_list):  # TODO: change ref layer, and other layers
        if idx == len(style_loss_list) - 1:  # last layer
            i.mode = 'capture_style_ref'
        else:
            i.mode = 'capture_style_others'
    net(style_img)

    tmp_ref_corr = None
    for idx, i in reversed(list(enumerate(style_loss_list))):  # TODO: change ref layer, and other layers
        if not i.mode == 'capture_style_ref':
            i.set_ref_infor(tmp_ref_corr)
        else:
            tmp_ref_corr = i.get_ref_infor()
            i.mode = 'None'
    net(style_img)

    if verbose:
        time_elapsed = time.time() - start_time
        print('@ Time Spend : {:.04f} m {:.04f} s'.format(time_elapsed // 60, time_elapsed % 60))

    # Histogram Loss 
    for i in histogram_loss_list:
        i.mode = 'capture_style'
    net(style_img)

    for i in histogram_loss_list:
        i.mode = 'capture_inter'
    net(inter_img)

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
    content_img, style_img, inter_img, tight_mask, loss_mask = preprocess(cfg, dtype, device)

    # Build Network 
    content_loss_list, style_loss_list, tv_loss_list, histogram_loss_list, net = build_net(cfg, device, loss_mask, StyleLossPass2, ContentLoss, TVLoss, HistogramLoss, cfg.verbose)

    # Capture FM & Compute Match 
    capture_fm_pass2(content_loss_list, style_loss_list, tv_loss_list, histogram_loss_list, inter_img, content_img, style_img, net, cfg.verbose)

    # Training 
    final_result = train(cfg, device, net, content_loss_list, style_loss_list, tv_loss_list, histogram_loss_list, start_img=inter_img, mask=loss_mask, verbose=cfg.verbose)
    
    # End Log 
    end_log(orig_stdout)


if __name__ == '__main__':
    main()
