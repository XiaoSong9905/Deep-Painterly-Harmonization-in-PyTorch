# EECS 442 Final Project @ UMich 
# No Commercial Use Allowed 

# This file is organized in a structure below 
# 1. calling this file directely `python auxilary_net.py` will train the network with choice of [start, resume] represent start to train from start / resume training 
# 2. calling the inference function of this file will load the model specified in the argument and evaluate the input style image 

import os 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torchvision.transforms as transforms 
import torchvision.models as models
import torchvision
import torch.nn.functional as F 
from PIL import Image 
import argparse
import copy
import math 
import numpy as np 
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt


def build_net(device, mode='start', model_file_path=None):
   '''
   Input : 
      mode : `start` start to train network with weight getten by imagenet 
             `continue` continue training network with model weight given in file_path 
             `eval` evaluate model output, an extra softmax layer will be added 
   Process:
      resnet18 is used instead of vgg16 like the original paper 
   Return :
      model with / without softmax layer depend on the mode choice 
         if `start` / `continue`, no softmax layer will be added, since loss is computed using CrossEntropy
         if `eval`, softmax layer will be added
   '''
   if mode == 'start':
      model = models.resnet18(pretrained=True)
      model.fc = nn.Linear(in_features=512, out_features=27, bias=True)

      # or you can use other initialization / model's default initialization 
      torch.nn.init.normal_(model.fc.weight.data, 0, 1)
      torch.nn.init.normal_(model.fc.bias.data, 0, 1)
      
   elif mode == 'continue' or mode == 'eval':
      model = models.resnet18(pretrained=False)
      model.fc = nn.Linear(in_features=512, out_features=27, bias=True)
      
      assert(model_file_path is not None)
      
      user_state_dict = torch.load(model_file_path)
      net_state_dict = model.state_dict()
      user_state_dict = {k: v for k, v in user_state_dict.items() if k in net_state_dict}
      net_state_dict.update(user_state_dict)
      model.load_state_dict(net_state_dict)

      if mode=='eval':
         model = nn.Sequential(model, nn.Softmax())
         model = model.eval()
         for param in model.parameters():
            param.requires_grad = False
   else:
      # TODO raise exception error 
      model = None 
   
   model = model.to(device)

   return model 


'''

使用vgg构建一个网络
在网络的上面定义一层loss 
定义一个trainer 

如果train 就 
     def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If your want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        losses = sum(loss for loss in loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()
 
 如果eval 
    使用model,eval / 或者不apply loss， 只输出weighted score


总结： model.train() 和 model.eval() 一般在模型训练和评价的时候会加上这两句，主要是针对由于model 在训练时和评价时 Batch Normalization 和 Dropout 方法模式不同；因此，在使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval；



'''