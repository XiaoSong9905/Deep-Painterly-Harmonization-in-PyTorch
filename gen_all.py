# Take reference from original source code of paper 
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
            print('Working on image idx = ', idx)
            part_cmd1 =' python3 pass1.py '\
					   ' -content_image data/' + str(idx) + '_naive.jpg  '\
					   ' -style_image   data/' + str(idx) + '_target.jpg '\
					   ' -tight_mask    data/' + str(idx) + '_c_mask.jpg '\
					   ' -dilated_mask  data/' + str(idx) + '_c_mask_dilated.jpg '\
                       ' -inter_image   official_result/' + str(idx) + '_inter_res.png' \
                       ' -gpu 0 ' \
					   ' -output_img    output/' + str(idx) + '_inter_res.jpg'\
                       ' -output_img_size 710' \
                       ' -n_iter 1300 ' \
					   ' -print_interval 100 -save_img_interval 100 &&'
            cmd = cmd + part_cmd1
    
    cmd = cmd[1:len(cmd)-1]
    print('#',cmd)
    os.system(cmd)


'''
            part_cmd2 =' th neural_paint.lua '\
					   ' -content_image data/' + str(idx) + '_naive.jpg '\
					   ' -style_image   data/' + str(idx) + '_target.jpg '\
					   ' -tmask_image   data/' + str(idx) + '_c_mask.jpg '\
					   ' -mask_image    data/' + str(idx) + '_c_mask_dilated.jpg '\
					   ' -cnnmrf_image  results/' + str(idx) + '_inter_res.jpg  '\
					   ' -gpu ' + str(j-1) + ' -original_colors 0 -image_size 700 '\
					   ' -index ' + str(idx) + ' -wikiart_fn data/wikiart_output.txt '\
					   ' -output_image  results/' + str(idx) + '_final_res.jpg' \
					   ' -print_iter 100 -save_iter 100 '\
					   ' -num_iterations 1000 &&' 
			cmd = cmd + part_cmd1 + part_cmd2
'''