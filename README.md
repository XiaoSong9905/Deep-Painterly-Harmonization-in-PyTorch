# Project : Deep Painterly Harmonization in PyTorch 

> EECS 442 Final Project at University of Michigan, Ann Arbor
> 
> Reimplementatino of paper `Deep Painterly Harmonization` 



Pytorch implementation of paper "[Deep Painterly Harmonization](https://arxiv.org/abs/1804.03189)"  


Official code written in Torch and lua can be found here [Github Link](https://github.com/luanfujun/deep-painterly-harmonization)


This PyTorch implementation follow the structure of [Neural Style Pt Github Link](https://github.com/jcjohnson/neural-style) by Justin Johnson where the network is first build and feature map is captured after the architrcture is build. In the original code [Official Code Github Link](https://github.com/luanfujun/deep-painterly-harmonization), the feature map is captured during the build of architecture which cause waist of computation. Also, the loss in different layer back prop by simply adding them up and call `loss_total.backward()` where in the offitial code, a backward hook is build to pass the loss gradient (`function StyleLoss:updateGradInput`). Remember some of the implementation difference from the official code is that Torch do not come with autograd but PyTorch internally come with autograde


**This Repo is still under active develop and have not yet finish**


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

14. 存下来pass1现在有的问题，然后在progress report上说找到了问题

15. pass2 rrelu5_1 是否有被用到， 把5_1的styel module删除掉

16. pass1 参考源代码，更改match，解决产生noise的问题 (pass1 根据源代码 'patchmatch_r_conv_kernal' line 1228 cuda_utils.cu  )

17. pass2 参考源代码，解决模糊的问题， 可能和match相关

18. 拆分pass1, pass2为两个文件

19. optimizer 不一样

20. pass1中是否有histogram loss / total variance loss

21. 存储float 0-255 图片的问题，需要试验一下,应该是没有问题的

## Getting Started with the code 

* Dependency 

```shell
torch >= 1.3.1 
torhchvision >= 0.4.1 
```

* Run on local computer with default setting 

By default

1. output image is stored under `output` directory 
2. output image shape is 64 
3. `0_target.jpg`, `0_naive.jpg` is used 

```python
python3 main.py
```

* Run on GPU setting like google colab 

```python
python3 main.py -gpu 0
```

* Run with specified style and content image 

```python
python3 main.py -style_image ./data/1_target.jpg \
        -native_image ./data/1_naive.jpg \
        -tight_mask data/1_c_mask.jpg \
        -dilated_mask data/1_c_mask_dilated.jpg
```

* Run with specified iteration, lr, etc 

```python
python3 main.py -lr 1e-1 -p1_n_iters 2000 -p2_n_iters 1000 
```

For more information on how to specify training process, check `main.py -> get_args()` 



 $L_{tv} = w_t \times \left(\sum_{c=1}^3\sum_{i=1}^{H-1}\sum_{j=1}^{W} (x_{i+1,j,c} - x_{i,j,c})^2 + \sum_{c=1}^3\sum_{i=1}^{H}\sum_{j=1}^{W - 1} (x_{i,j+1,c} - x_{i,j,c})^2\right)$



## Auxilary Network Dataset

Some dataset have been found that might work for the auxilary network (network that pick the weight of loss)

[WikiArt from ArtGan - Github Link](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset)

* [dataset download link](http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip)

* [annotation download link](http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart_csv.zip)  (use the style data part)

