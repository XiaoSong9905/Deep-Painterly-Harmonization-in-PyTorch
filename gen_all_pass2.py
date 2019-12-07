# Take reference from https://github.com/DmitryUlyanov/deep-image-prior 
import os
import math

numImgs = 35
# numGpus = 16 
numGpus = 1

if os.path.exists('output') == 0:
	os.mkdir('output')

N = int(math.ceil(float(numImgs)/numGpus))

for j in range(1, numGpus+1):
	cmd = ''
	for i in range(1, N+1):
		idx = (i-1) * numGpus + (j-1)
		if idx >= 0 and idx < numImgs:
			if idx in [0, 2, 4, 9, 14, 26, 18, 24]:
				print('Working on image idx = ', idx)
				part_cmd2 =' python3 pass2.py '\
						' -content_image data/' + str(idx) + '_naive.jpg  '\
						' -style_image   data/' + str(idx) + '_target.jpg '\
						' -tight_mask    data/' + str(idx) + '_c_mask.jpg '\
						' -dilated_mask  data/' + str(idx) + '_c_mask_dilated.jpg '\
						' -inter_image   output/' + str(idx) + '_inter_res.jpg' \
						' -gpu 0 ' \
						' -output_img    output/' + str(idx) + '_final_res.jpg'\
						' -output_img_size 710' \
						' -n_iter 3000 ' \
						' -lr 3e-1 ' \
						' -style_layers relu1_1,relu2_1,relu3_1,relu4_1 '\
						' -content_layers relu4_1 ' \
						' -print_interval 100 -save_img_interval 10 && '
			else:
				part_cmd2 =' python3 pass2.py '\
						' -content_image data/' + str(idx) + '_naive.jpg  '\
						' -style_image   data/' + str(idx) + '_target.jpg '\
						' -tight_mask    data/' + str(idx) + '_c_mask.jpg '\
						' -dilated_mask  data/' + str(idx) + '_c_mask_dilated.jpg '\
						' -inter_image   output/' + str(idx) + '_inter_res.jpg' \
						' -gpu 0 ' \
						' -output_img    output/' + str(idx) + '_final_res.jpg'\
						' -output_img_size 710' \
						' -n_iter 2000 ' \
						' -lr 3e-1 ' \
						' -style_layers relu1_1,relu2_1,relu3_1,relu4_1 '\
						' -content_layers relu4_1 ' \
						' -print_interval 100 -save_img_interval 100 && '
			cmd = cmd + part_cmd2
	cmd = cmd[1:len(cmd)-1]
	print('#',cmd)
	os.system(cmd)
