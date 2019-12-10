# Take reference from https://github.com/DmitryUlyanov/deep-image-prior 
import os
import math

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-idx", type=int, default=0)
parser.add_argument("-p", type=int, default=2)
parser.add_argument("-size", type=int, default=512)
cfg = parser.parse_args()
idx = cfg.idx
pass12 = cfg.p
size = cfg.size

if not os.path.exists('output'):
    os.mkdir('output')

if pass12 == 1:
    cmd = ' python3 pass1.py '\
        ' -content_image data/' + str(idx) + '_naive.jpg  '\
        ' -style_image   data/' + str(idx) + '_target.jpg '\
        ' -tight_mask    data/' + str(idx) + '_c_mask.jpg '\
        ' -dilated_mask  data/' + str(idx) + '_c_mask_dilated.jpg '\
        ' -inter_image   official_result/' + str(idx) + '_inter_res.jpg' \
        ' -gpu 0 ' \
        ' -output_img    output/' + str(idx) + '_inter_res.jpg'\
        ' -output_img_size ' + str(size) + ' ' \
        ' -n_iter 3000 ' \
        ' -lr 3e-1 ' \
        ' -print_interval 100 -save_img_interval 10 '\
        ' -verbose '
elif pass12 == 2:
    cmd =' python3 pass2.py '\
        ' -content_image data/' + str(idx) + '_naive.jpg  '\
        ' -style_image   data/' + str(idx) + '_target.jpg '\
        ' -tight_mask    data/' + str(idx) + '_c_mask.jpg '\
        ' -dilated_mask  data/' + str(idx) + '_c_mask_dilated.jpg '\
        ' -inter_image   output/' + str(idx) + '_inter_res.jpg' \
        ' -gpu 0 ' \
        ' -output_img    output/' + str(idx) + '_final_res.jpg'\
        ' -output_img_size ' + str(size) + ' '\
        ' -n_iter 3000 ' \
        ' -lr 1.5e-1 ' \
        ' -style_layers relu1_1,relu2_1,relu3_1,relu4_1 '\
        ' -content_layers relu4_1 ' \
        ' -histogram_layers relu1_1,relu4_1 ' \
        ' -histogram_weight 1 ' \
        ' -style_weight 1e-4 ' \
        ' -content_weight 3e-1 ' \
        ' -tv_weight 0.1 '\
        ' -print_interval 10 -save_img_interval 10  ' \
        ' -verbose '

print('#',cmd)
os.system(cmd)
