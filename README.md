# Project : Deep Painterly Harmonization in PyTorch 

> EECS 442 Final Project at University of Michigan, Ann Arbor
> 
> Reimplementatino of paper `Deep Painterly Harmonization` 



Pytorch implementation of paper "[Deep Painterly Harmonization](https://arxiv.org/abs/1804.03189)"  


Official code written in Torch and lua can be found here [Github Link](https://github.com/luanfujun/deep-painterly-harmonization)


This PyTorch implementation follow the structure of [Neural Style Pt Github Link](https://github.com/jcjohnson/neural-style) by Justin Johnson where the network is first build and feature map is captured after the architrcture is build. In the original code [Official Code Github Link](https://github.com/luanfujun/deep-painterly-harmonization), the feature map is captured during the build of architecture which cause waist of computation. Also, the loss in different layer back prop by simply adding them up and call `loss_total.backward()` where in the offitial code, a backward hook is build to pass the loss gradient (`function StyleLoss:updateGradInput`). Remember some of the implementation difference from the official code is that Torch do not come with autograd but PyTorch internally come with autograde


**This Repo is still under active develop and have not yet finish**


## TODO 

1. StyleLossPass2 

2. (sx) StyleLossPass1 [DONE]

3. (sx) Pass 1 debug & test validation [DONE]

4. pass 2 (should be similar to pass 1 except broadcating the match relation between different layer)

5. (sx) notebook for bp work on mask area [DONE]

6. (sx) periodic save, periodic print [DONE]

7. (sx) need to check in the original lua code, what does `dG:div(msk:sum())` is doing, why divide the gradient, how can this be acapted into out code. [DONE]

8. TVLoss module 

9. code update to support GPU setting, some job is done but not all. need to run on google colab to check GPU support 


## Auxilary Network Dataset 

Some dataset have been found that might work for the auxilary network (network that pick the weight of loss)

[WikiArt from ArtGan - Github Link](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset)

* [dataset download link](http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip)

* [annotation download link](http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart_csv.zip)  (use the style data part)


[Kaggle Competation on Art style](https://www.kaggle.com/c/painter-by-numbers/data)