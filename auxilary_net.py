# EECS 442 Final Project @ UMich 
# No Commercial Use Allowed 

# This file is organized in a structure below 
# 1. calling this file directely `python auxilary_net.py` will train the network with choice of [start, resume] represent start to train from start / resume training 
# 2. calling the inference function of this file will load the model specified in the argument and evaluate the input style image 

import os 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.optim import lr_scheduler
import torchvision.transforms as transforms 
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision
import torch.nn.functional as F 
from PIL import Image 
import argparse
import copy
import math 
import numpy as np 
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import time 

from utils import * 

parser = argparse.ArgumentParser()

parser.add_argument("-mode", help="mode for training start / resume", choices=['start', 'resume'], default='start')
parser.add_argument("-model_file_path", help="model weight file path if mode is resume", default=None)
parser.add_argument("-data_dir", help="directory path to dataset", default='./ArtStyleData/data')
parser.add_argument("-ann_fiile", help="file path to dataset", default='./ArtStyleData/annotation/style_train.csv')

parser.add_argument("-lr", type=float, default=1e-1)
parser.add_argument("-epoch", type=int, default=10)
parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-momentum", type=int, default=32)

parser.add_argument("-print_interval", type=int, default=100)
parser.add_argument("-save_model_interval", type=int, default=100)
parser.add_argument("-save_model_path", help="path to save model", default="./auxilary_model/")
parser.add_argument("-gpu", help="Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1", default=-1)

cfg = parser.parse_args()

def build_net(mode='start', model_file_path=None):
   '''
   Input : 
      mode : `start` start to train network with weight getten by imagenet 
             `continue` continue training network with model weight given in file_path 
             `eval` evaluate model output, an extra softmax layer will be added 
      device : if this fucntion is called by inference(), then the device type of the `main.py` program will be use 
               if this function is called by train(), then the device type of this file will be used 
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

      print('Model with ImageNet weight is build')

   elif mode == 'continue' or mode == 'eval':
      model = models.resnet18(pretrained=False)
      model.fc = nn.Linear(in_features=512, out_features=27, bias=True)
      
      assert(model_file_path is not None)
      
      user_state_dict = torch.load(model_file_path)
      net_state_dict = model.state_dict()
      user_state_dict = {k: v for k, v in user_state_dict.items() if k in net_state_dict}
      net_state_dict.update(user_state_dict)
      model.load_state_dict(net_state_dict)

      print('Model with weight in `{}` is build'.format(model_file_path))

      if mode=='eval':
         model = nn.Sequential(model, nn.Softmax())
         model = model.eval()
         for param in model.parameters():
            param.requires_grad = False
   else:
      # TODO raise exception error 
      model = None 
   
   return model 

def train_net():
   # Mostly follow the transfer learning step from here 
   # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html 

   dtype, device = setup(cfg)

   # Get Data 
   print('===> Start Prepare Data')
   start_time = time.time()

   data_transforms = {
      'train': transforms.Compose([
         transforms.RandomResizedCrop(300),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # TODO recheck this normalization parameter 
      ]),
      'val': transforms.Compose([
         transforms.Resize(300),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
   }

   # Each Type of data is stored in corresponding folder, this is not true for all the data you may get but is true in this case 
   image_datasets = {x: datasets.ImageFolder(os.path.join(cfg.data_dir, x),
                                             data_transforms[x]) for x in ['train', 'val']}
   dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=cfg.batch_size,
                                                shuffle=True, num_workers=4) for x in ['train', 'val']}
   dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
   class_names = image_datasets['train'].classes

   print('Corresponding relationship between name and idx is :', image_datasets['train'].class_to_idx)
   print('===> Finish Prepare Data with {}min {} second'.format(str( (time.time()-start_time)//60 ), (time.time()-start_time)%60 ) )

   # Get Model & Optimizer & Schedular 
   print('===> Start Training Network')
   start_time = time.time()

   model = build_net(mode=cfg.mode, model_file_path=cfg.model_file_path)
   model = model.to(device)

   optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
   schedular = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

   criterian = nn.CrossEntropyLoss()

   # Run epoch 
   # Model performence is evaluated every epoch 
   for epoch in range(cfg.epoch):
      print('Epoch {}/{}'.format(epoch, cfg.epoch - 1))




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