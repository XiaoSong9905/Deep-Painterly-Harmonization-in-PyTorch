# Take reference from https://github.com/DmitryUlyanov/deep-image-prior
import os
import math

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-idx", type=int, default=0)
parser.add_argument("-p", type=int, default=2)
parser.add_argument("-size", type=int, default=512)
parser.add_argument("-patch", type=int, default=6)
cfg = parser.parse_args()
idx = cfg.idx
pass12 = cfg.p
size = cfg.size
patch = cfg.patch

if not os.path.exists('outputour'):
    os.mkdir('outputour')

pre = '_our'

if pass12 == 1:
    cmd = ' python3 pass1.py '\
        ' -content_image ourdata/' + str(idx) + pre + '_naive.jpg  '\
        ' -style_image   ourdata/' + str(idx) + pre + '_target.jpg '\
        ' -tight_mask    ourdata/' + str(idx) + pre + '_c_mask.png '\
        ' -dilated_mask  ourdata/' + str(idx) + pre + '_c_mask_dilated.png '\
        ' -inter_image   ourdata/inter_dummy.jpg' \
        ' -gpu 0 ' \
        ' -output_img    outputour/' + str(idx) + pre + '_inter_res.jpg'\
        ' -output_img_size ' + str(size) + ' ' \
        ' -n_iter 3000 ' \
        ' -lr 3e-1 ' \
        ' -match_patch_size ' + str(patch) + ' '\
        ' -print_interval 100 -save_img_interval 10 '\
        ' -verbose '
elif pass12 == 2:
    cmd =' python3 pass2.py '\
        ' -content_image ourdata/' + str(idx) + pre + '_naive.jpg  '\
        ' -style_image   ourdata/' + str(idx) + pre + '_target.jpg '\
        ' -tight_mask    ourdata/' + str(idx) + pre + '_c_mask.png '\
        ' -dilated_mask  ourdata/' + str(idx) + pre + '_c_mask_dilated.png '\
        ' -inter_image   outputour/' + str(idx) + pre + '_inter_res.jpg' \
        ' -gpu 0 ' \
        ' -output_img    outputour/' + str(idx) + pre + '_final_res.jpg'\
        ' -output_img_size ' + str(size) + ' '\
        ' -n_iter 3000 ' \
        ' -lr 3e-1 ' \
        ' -style_layers relu1_1,relu2_1,relu3_1,relu4_1 '\
        ' -content_layers relu4_1 ' \
        ' -histogram_layers relu1_1,relu4_1 ' \
        ' -histogram_weight 1' \
        ' -style_weight 1e-4 ' \
        ' -content_weight 3e-1 ' \
        ' -tv_weight 0.005 '\
        ' -print_interval 10 -save_img_interval 10  ' \
        ' -verbose '

print('#',cmd)
os.system(cmd)
