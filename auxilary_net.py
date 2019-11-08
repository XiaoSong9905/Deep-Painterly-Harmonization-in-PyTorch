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
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F 
from PIL import Image 
import argparse
import copy
import math 
import numpy as np 
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import time 
import pandas as pd 

from utils import * 

parser = argparse.ArgumentParser()

parser.add_argument("-mode", help="mode for training start / resume", choices=['start', 'resume'], default='start')
parser.add_argument("-checkpoint_file", help="checkpoint pass if mode=resume", default=None)
parser.add_argument("-data_dir", help="directory path to dataset", default='./ArtStyleData/data')
parser.add_argument("-ann_fiile", help="relative file path to dataset", default='./ArtStyleData/annotation/style_train.csv')

parser.add_argument("-lr", type=float, default=1e-1)
parser.add_argument("-epoch", type=int, default=10)
parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-momentum", type=int, default=32)

parser.add_argument("-save_model_interval", type=int, default=100)
parser.add_argument("-save_checkpoint_path", help="path to save model", default="./auxilary_model/")
parser.add_argument("-gpu", help="Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1", default=-1)

parser.add_argument("-debug_mode", type=bool, default=True)

cfg = parser.parse_args()

def build_net_optim_schedu(cfg, mode='start', checkpoint_file=None):
   '''
   Input : 
      mode : `start` start to train network with weight getten by imagenet 
             `continue` continue training network with model weight given in file_path 
             `inference` evaluate model output, an extra softmax layer will be added 
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

      # Build Model 
      print('Model with ImageNet weight is build')
      model = models.resnet18(pretrained=True)
      model.fc = nn.Linear(in_features=512, out_features=27, bias=True)

      #torch.nn.init.normal_(model.fc.weight.data, 0, 1)
      #torch.nn.init.normal_(model.fc.bias.data, 0, 1)

      # Build Optimizer 
      optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)

      # Build Schedular 
      schedular = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

      user_epoch = 0

   elif mode == 'continue' or mode == 'inference':
      
      # Build Model 
      assert(checkpoint_file is not None)
      print('Model with checkpoint `{}` is build'.format(checkpoint_file))

      model = models.resnet18(pretrained=False)
      model.fc = nn.Linear(in_features=512, out_features=27, bias=True)
      
      checkpoint = torch.load(checkpoint_file)
      user_model_state_dict = checkpoint['model']
      user_epoch = checkpoint['epoch']
      user_optim_state_dict = checkpoint['optimizer']

      # Fault tolerence model load 
      net_state_dict = model.state_dict()
      user_model_state_dict = {k: v for k, v in user_model_state_dict.items() if k in net_state_dict}
      net_state_dict.update(user_model_state_dict)
      model.load_state_dict(net_state_dict)

      if mode=='inference':
         model = nn.Sequential(model, nn.Softmax(dim=0))
         model = model.eval()
         for param in model.parameters():
            param.requires_grad = False
            
         optimizer = None 
         schedular = None 

      else:
         # Build optimizer 
         optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
         optimizer.load_state_dict(user_optim_state_dict)
         
         # Build Schedular 
         schedular = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

   # Build Optimizer 


   return model, optimizer, schedular, user_epoch

class ArtDataset(Dataset):
    def __init__(self, csv_file, data_root_dir, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        # self.dataframe['file'][idx] : Impressionism/edgar-degas_dancers-on-set-1880.jpg
        self.data_root_dir = data_root_dir
        self.transform = transform # depend on argument, either transform for train or val 
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        '''
        Get lable and image for `idx \in [0, self.__len__]
        '''
        img_file = os.path.join(self.data_root_dir, self.dataframe['file'][idx])
        img = Image.open(img_file)
        lable = self.dataframe['cat'][idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, lable 
        

def train_net():
   # Mostly follow the transfer learning step from here 
   # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html 

   dtype, device = setup(cfg)

   # Build Network 
   model, optimizer, schedular, start_epoch = build_net_optim_schedu(cfg, mode=cfg.mode, checkpoint_file=cfg.checkpoint_file)
   model = model.to(device)

   criterian = nn.CrossEntropyLoss()

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

   train_dataset = ArtDataset(cfg.ann_fiile, cfg.data_dir, data_transforms['train'])
   val_dataset = ArtDataset(cfg.ann_fiile, cfg.data_dir, data_transforms['val'])

   train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
   val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

   dataset_sizes = {'train':len(train_dataset), 'val':len(val_dataset)}

   print('===> Finish Prepare Data with {}min {} second'.format(str( (time.time()-start_time)//60 ), (time.time()-start_time)%60 ) )
   
   # Get Model & Optimizer & Schedular 
   print('===> Start Training Network')
   start_time = time.time()

   # Run epoch 
   # Model performence is evaluated every epoch 
   end_epoch = cfg.epoch
   for epoch in range(start_epoch, end_epoch): # TODO change the epoch here to support training from middle 

      # Train 
      schedular.step()
      model.train()

      training_loss = 0.0 # For whole training dataset 
      training_acc = 0.0 
      
      # Run iterator 
      for i, (inputs, lables) in enumerate(train_dataloader):
         optimizer.zero_grad()

         inputs = inputs.to(device)
         lables = lables.to(device)
         
         outputs = model(inputs)
         loss = criterian(outputs, lables)
         _, preds = torch.max(outputs, 1)

         import pdb; pdb.set_trace()
         print('lables shape', lables.shape)
         print('outputs shape', outputs.shape)
         print('inputs shape', inputs.shape)

         loss.backward()
         optimizer.step()

         training_loss += loss.item() * inputs.shape[0]
         training_acc += torch.sum( preds == lables.data )

      
      training_loss = training_loss / dataset_sizes['train']
      training_acc = training_acc / dataset_sizes['train']

      # Eval 
      # TODO see discussion notebook and see hoe to save gradient 
      model.eval()
      optimizer.zero_grad()

      val_loss = 0.0 # For whole training dataset 
      val_acc = 0.0 
      
      # Run iterator 
      for i, (inputs, lables) in enumerate(val_dataloader):
         inputs = inputs.to(device)
         lables = lables.to(device)
         
         outputs = model(inputs)
         loss = criterian(outputs, lables)
         _, preds = torch.max(outputs, 1)

         val_loss += loss.item() * inputs.shape[0]
         val_acc += torch.sum( preds == lables.data )

      
      val_loss = val_loss / dataset_sizes['val']
      val_acc = val_acc / dataset_sizes['val']

      print('Epoch {}/{}; Train Loss {} Train Acc {} ;Val Loss {} Val Acc {}'.format(epoch, \
         cfg.epoch - 1, training_loss, training_acc, val_loss, val_acc))


      # Save Model if required 
      if epoch % cfg.save_model_interval == 0 or epoch == end_epoch-1:
         state = {'epoch': epoch + 1, 
                  'model':model.state_dict()}
         save_file_name = 'epoch_'+str(epoch)+'_acc_'+str(val_acc)
         torch.save(state, save_file_name)

   print('===> Finish Train Network with {} min {} second'.format(str( (time.time()-start_time)//60 ), (time.time()-start_time)%60 ) )

def inference(device, img, checkpoint_file):
   model, _ , _, _= build_net_optim_schedu(cfg=None, mode='inference', checkpoint_file=checkpoint_file)
   model = model.to(device)
   
   output = model(img)

   output_np = output.squeeze(0).numpy()

   # TODO add weight based on paper 
   style_loss_weight = np.array(([[10]*27])).T.squeeze(1)
   tv_loss_weight = np.array(([[10]*27])).T.squeeze(1) # Need shape to be (27,)

   assert(style_loss_weight.shape == output_np.shape)
   assert(tv_loss_weight.shape == output_np.shape)

   style_loss_weight_final = np.sum(np.multiply(output_np, style_loss_weight))
   tv_loss_weight_final = np.sum(np.multiply(output_np, tv_loss_weight))

   print('===> Style Image loss weight choice {} & {}'.format(style_loss_weight_final, tv_loss_weight_final))

   return style_loss_weight_final, tv_loss_weight_final

if __name__ == '__main__':
   train_net()
   #img = torch.rand((1, 3, 300, 300))
   #inference('cpu', img, checkpoint_file='./initial_model')