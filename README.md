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

23. what's left to do : 如何解决noise， 一个段落 [DONE]

24. Histogram Loss 

25. Pass2 跑数据

26. 完成poster，打印poster

27. Final Report 

28. 非论文的数据

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


* Run on full detailed mode (if not specify, the program will run in silence mode) 

```shell
python3 pass1.py -verbose

python3 pass2.py -verbose
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

* `verbose` : if not specify, then the program will run in silence mode



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

