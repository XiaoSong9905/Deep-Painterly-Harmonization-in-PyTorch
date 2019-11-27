# Deep Painterly Harmonization in PyTorch

> EECS 442 Final Project at University of Michigan, Ann Arbor
> 
> Reimplementation of paper `Deep Painterly Harmonization` 



Pytorch implementation of paper "[Deep Painterly Harmonization](https://arxiv.org/abs/1804.03189)"  


Official code written in Torch and lua can be found here [Github Link](https://github.com/luanfujun/deep-painterly-harmonization)

This PyTorch implementation follow the structure of [Neural Style Pt Github Link](https://github.com/jcjohnson/neural-style)



**Pass1 of update is finished, try it out**

**Pass2 of update have not yet finish and still under active development**



## TODO

1. (dd, zw) StyleLossPass2 [DONE]

2. (sx) StyleLossPass1 [DONE]

3. (sx) Pass 1 debug & test validation [DONE]

4. (zw, dd) Pass 2 (should be similar to pass 1 except broadcating the match relation between different layer)

5. (sx) notebook for bp work on mask area [DONE]

6. (sx) periodic save, periodic print [DONE]

7. (sx) need to check in the original lua code, what does `dG:div(msk:sum())` is doing, why divide the gradient, how can this be acapted into out code. [DONE]

8. (zw)TVLoss module [DONE]

9. code update to support GPU setting, some job is done but not all. need to run on google colab to check GPU support 

10. (sx) Auxilary network for style loss build, need to rewrite [DONE]

11. (sx) train auxilary network [NEED TO CHECK WITH DAVID ON CONVERGENCE PROBLEM]

12. (sx) Supprot Tensorboard X for style transform and auxilary network 

13. (zw) Code formatting 

14. (sx) 存下来pass1现在有的问题，然后在progress report上说找到了问题 [DONE]

15. pass2 rrelu5_1 是否有被用到， 把5_1的styel module删除掉

16. (sx) pass1 参考源代码，更改match，解决产生noise的问题 (pass1 根据源代码 'patchmatch_r_conv_kernal' line 1228 cuda_utils.cu  )  [DONE]

17. pass2 参考源代码，解决模糊的问题， 可能和match相关

18. 拆分pass1, pass2为两个文件

19. (sx) optimizer 不一样  [DONE]

20. (sx) pass1中是否有histogram loss / total variance loss  [DONE]

21. (sx) 存储float 0-255 图片的问题，需要试验一下,应该是没有问题的  [DONE] 



## Requirement

* Dependency 

```shell
torch >= 1.3.1 
torhchvision >= 0.4.1 
python >= 3.7
```



## Useage

### Download Model Weight

```shell
python3 models/download_models.py
```



### Understand Notebook

Before getting started with the code, checkout `notebook/*` to understand some implementation detail 



### Run Code

* Run on Default Setting 

By default

1. output image is stored under `output` directory 
2. output image shape is 512 
3. `0_target.jpg`, `0_naive.jpg` is used 

```shell
python3 pass1.py

python3 pass2.py -output_img output/0_pass2_out.png
```



* Run Pass2 Starting from Offitial Pass1 result 

```shell
python3 pass2.py -output_img output/0_pass2_out.png  \
								 -native_image official_result/0_inter_res.jpg
```



* Run on GPU setting like google colab 

```shell
python3 pass1.py -gpu 0

python3 pass2.py -gpu 0
```



* Run with specified style and content image 

```shell
python3 pass1.py 
        -style_image ./data/1_target.jpg \
        -native_image ./data/1_naive.jpg \
        -tight_mask data/1_c_mask.jpg \
        -dilated_mask data/1_c_mask_dilated.jpg
```



* Run with specified iteration, lr, etc 

```shell
python3 pass1.py -lr 1e-1 -p1_n_iters 2000 -p2_n_iters 1000 
```



### Options

* `style_image`, `native_image`, `tight_mask`, `dilated_mask` : specify which style & content image to transfer 

* `output_img` : name for output image, defualt `output/0_pass1_out.png`

* `output_img_size` : output image size of the larger side 

* `optim`, `lr`, `n_iter` : learning parameter choice 

* `print_interval` : print loss interval 

* `save_img_interval` : save intermediate image interval 

* `gpu` : 'cpu' or '0'

* `content_layers`, `style_layers` : specify content layer, style layer 

* `content_weight`, `style_weight`, `tv_weight` : weight for each loss, all style loss share the same weight 

* `model` : model choice, 'vgg16' / 'vgg19'

* `model_file` : user specify weight for model, must match model choice 

* `match_patch_size` : patch size for feature map matching, default 3 like paper 

* `mask_on` : use mask or not, default mask on like paper, you can also choose mask off to compute mathing feature map for whole content image & compuet loss on whole content image 

* `log_on` : use log or not, default log off 

* `log_file` : file name to log 



## Auxilary Network Dataset

Some dataset have been found that might work for the auxilary network (network that pick the weight of loss)

[WikiArt from ArtGan - Github Link](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset)

* [dataset download link](http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip)

* [annotation download link](http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart_csv.zip)  (use the style data part)



## Citation 

If you find this code useful for your research, please cite:

```shell
@misc{DPH2018,
  author = {Xiao Song, Ziyan Wang, Deyang Dai},
  title = {Deep-Painterly-Harmonization-in-PyTorch},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/XiaoSong9905/Deep-Painterly-Harmonization-in-PyTorch}},
}
```

