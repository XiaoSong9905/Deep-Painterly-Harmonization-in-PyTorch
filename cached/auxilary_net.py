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
from PIL import Image, ImageFile
import argparse
import copy
import math 
import numpy as np 
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import time 
import pandas as pd 
from tensorboardX import SummaryWriter 

from utils import * 

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000                                                                                              

parser = argparse.ArgumentParser()

parser.add_argument("-mode", help="mode for training start / resume", choices=['start', 'resume'], default='start')
parser.add_argument("-checkpoint_file", help="checkpoint pass if mode=resume", default=None)
parser.add_argument("-data_dir", help="directory path to dataset", default='./ArtStyleData/data')
parser.add_argument("-train_ann_file", help="relative file path to dataset", default='./ArtStyleData/annotation/style_train.csv')
parser.add_argument("-val_ann_file", help="relative file path to dataset", default='./ArtStyleData/annotation/style_val.csv')

parser.add_argument("-lr", type=float, default=1e-3)
parser.add_argument("-epoch", type=int, default=10)
parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-momentum", type=int, default=0.9)

parser.add_argument("-print_iteration_interval", type=int, default=100)
parser.add_argument("-save_model_iter_interval", type=int, default=100)
parser.add_argument("-save_checkpoint_path", help="path to save model", default="./auxilary_model/")
parser.add_argument("-gpu", help="Zero-indexed ID of the GPU to use; for CPU mode set -gpu = cpu", default='cpu')

parser.add_argument("-debug_mode", type=bool, default=True)
parser.add_argument("-compute_dataset", type=bool, default=False)

cfg = parser.parse_args()

def build_net_optimizer_schedular(cfg, device, mode='start', checkpoint_file=None):
   '''
   Input : 
      mode : `start` start to train network with weight getten by imagenet 
             `continue` continue training network with model weight given in file_path 
             `inference` evaluate model output, an extra softmax layer will be added 
   Process:
      resnet18 is used instead of vgg16 like the original paper 
   Return :
      model with / without softmax layer depend on the mode choice 
         if `start` / `continue`, no softmax layer will be added, since loss is computed using CrossEntropy
         if `inference`, softmax layer will be added
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

   elif mode == 'resume' or mode == 'inference':
      
      # Build Model 
      assert(checkpoint_file is not None)
      print('Model with checkpoint `{}` is build'.format(checkpoint_file))

      model = models.resnet18(pretrained=False)
      model.fc = nn.Linear(in_features=512, out_features=27, bias=True)
      
      checkpoint = torch.load(checkpoint_file, map_location=device)
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
         #optimizer.load_state_dict(user_optim_state_dict)
      
         # Build Schedular 
         schedular = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

   return model, optimizer, schedular, user_epoch

class ArtDataset(Dataset):
    def __init__(self, csv_file, data_root_dir, transform=None):
        self.dataframe = pd.read_csv(csv_file, header=None, names=['file', 'cat'])
        # self.dataframe['file'][idx] : Impressionism/edgar-degas_dancers-on-set-1880.jpg
        self.data_root_dir = data_root_dir
        self.transform = transform # depend on argument, either transform for train or val 
        
        print('===> ArtDataset with csv file [{}] and data root [{}] build'.format(csv_file, data_root_dir))

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        '''
        Get lable and image for `idx \in [0, self.__len__]
        '''
        img_file = os.path.join(self.data_root_dir, self.dataframe['file'][idx])
        try:
           #print(img_file)
           img = Image.open(img_file)
        except Exception as e:
           print('dataset idx {} invalid'.format(idx))
           idx = idx - 1 if idx > 0 else idx + 1 
           return self.__getitem__(idx)

        lable = self.dataframe['cat'][idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, lable 

def compute_dataset_mean_std():
   '''
   Simplified version of finding dataset mean, std 

   `Var(X) = E[(X - E(X))^2] - E(X^2) - E(X)^2`
   `std = var**0.5`
   `E(X^2) = 1/N \sum_{i=0}^N (x_i)`
   '''

   transform = transforms.Compose([
         transforms.Resize((300, 300)),
         transforms.ToTensor()
   ])

   train_dataset = ArtDataset(cfg.train_ann_file, cfg.data_dir, transform)
   train_dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=False, num_workers=0) # On MACOS, set num_workers=0, on unix, set nun_workers=4

   mean = 0.0 
   meansq = 0.0 
   std = 0.0 

   print('===> Start Iteration')
   for i, (data, _ ) in enumerate(train_dataloader):
      data = data.view(data.shape[0], data.shape[1], -1)
      mean += data.mean(2).sum(0)
      meansq += (data.mean(2)**2).sum(0)

      if i % 2 == 0:
         print('Iteration {}'.format(i))

   mean = mean / len(train_dataset) 
   meansq = meansq / len(train_dataset) 
   std = (meansq - mean**2)**0.5 

   print('mean : ', mean, '; std : ', std)

def train_net():
   # Mostly follow the transfer learning step from here 
   # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html 

   dtype, device = setup(cfg)

   # Set up TensorboardX
   writer = SummaryWriter('log')

   # Build Network 
   model, optimizer, schedular, start_epoch = build_net_optimizer_schedular(cfg, device, mode=cfg.mode, checkpoint_file=cfg.checkpoint_file)

   criterian = nn.CrossEntropyLoss()

   # Get Data 
   print('===> Start Prepare Data')
   start_time = time.time()

   data_transforms = {
      'train': transforms.Compose([
         transforms.Resize((300, 300)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize( [0.5220, 0.4686, 0.4041],[0.1701, 0.1727, 0.1799]) 
      ]),
      'val': transforms.Compose([
         transforms.Resize((300, 300)),
         transforms.ToTensor(),
         transforms.Normalize( [0.5220, 0.4686, 0.4041],[0.1701, 0.1727, 0.1799])
      ]),
   }

   train_dataset = ArtDataset(cfg.train_ann_file, cfg.data_dir, data_transforms['train'])
   val_dataset = ArtDataset(cfg.val_ann_file, cfg.data_dir, data_transforms['val'])

   train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0) # On MACOS, set num_workers=0, on unix, set nun_workers=4
   val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

   dataset_sizes = {'train':len(train_dataset), 'val':len(val_dataset)}
   
   time_elapsed = time.time() - start_time
   print('===> Finish Prepare Data with time{:.0f}m {:.04f}s'.format(time_elapsed // 60, time_elapsed % 60))

   # Get Model & Optimizer & Schedular 
   print('===> Start Training Network')
   start_time = time.time()
   model = model.to(device)

   # Run epoch 
   # Model performence is evaluated every epoch 
   num_epoch = cfg.epoch
   for epoch in range(start_epoch, start_epoch+num_epoch): 
      # Train 
      model.train()
      training_loss = 0.0 # For whole training dataset 
      training_acc = 0.0 
      
      # Run iterator 
      for i, (inputs, lables) in enumerate(train_dataloader):
         niter = epoch * len(train_dataloader) + i 

         optimizer.zero_grad()

         inputs = inputs.to(device)
         lables = lables.to(device)
         
         outputs = model(inputs)
         loss = criterian(outputs, lables)
         _, preds = torch.max(outputs, 1)

         #print('lables shape', lables.shape)
         #print('outputs shape', outputs.shape)
         #print('inputs shape', inputs.shape)
         # lables shape torch.Size([32])
         # outputs shape torch.Size([32, 27])
         # inputs shape torch.Size([32, 3, 300, 300])

         loss.backward()
         optimizer.step()

         training_loss += loss.item() * inputs.shape[0]
         training_acc += torch.sum( preds == lables.data )

         if i % cfg.print_iteration_interval == 0:
            print('Epoch {:2d} Iteration {:4d} loss {:.05f}'.format(epoch, i, loss.item()))
         
         # Record train loss for tensorboard x
         if i % 10 == 0:
            writer.add_scalar('Train/Loss', loss.item(), niter)

         # Save model per inteval 
         if niter % cfg.save_model_iter_interval == 0:
            state = {'epoch': epoch, 
                     'iter': i+1,
                     'model':model.state_dict(), 
                     'optimizer':optimizer.state_dict()}
            save_file_name = 'epoch_'+str(epoch)+'_iter_'+str(i)+'_loss_'+str(loss.item())
            torch.save(state, save_file_name)

      training_loss = training_loss / dataset_sizes['train']
      training_acc = training_acc / dataset_sizes['train']

      writer.add_scalar('Train/Epoch_Acc', training_acc, epoch)

      schedular.step()

      # Eval 
      model.eval()

      val_loss = 0.0 # For whole training dataset 
      val_acc = 0.0 
      
      with torch.no_grad():
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

      # Tensorboard X  
      writer.add_scalar('Test/Epoch_Acc', val_acc, epoch)
      writer.add_scalar('Test/Epoch_Loss', val_loss, epoch)

      time_elapsed = time.time() - start_time
      print('Epoch {}/{}; Train Loss {:.05f} Train Acc {:.05f} ;Val Loss {:.05f} Val Acc {:.05f} Time {:.0f}m {:.04f}s'.format(epoch, \
         cfg.epoch - 1, training_loss, training_acc, val_loss, val_acc, time_elapsed // 60, time_elapsed % 60))

      # Save Model every epoch 
      state = {'epoch': epoch + 1, 
               'iter': -1,
               'model':model.state_dict(), 
               'optimizer':optimizer.state_dict()}
      save_file_name = 'epoch_'+str(epoch)+'_iter_'+str(i)+'_acc_'+str(val_acc.item())
      torch.save(state, save_file_name)

   time_elapsed = time.time() - start_time
   print('===> Finish Train Network with time {:.0f}m {:.04f}s'.format(time_elapsed // 60, time_elapsed % 60))

   # Save Model after training 
   state = {'epoch': epoch + 1, 
            'iter': -1,
            'model':model.state_dict(), 
            'optimizer':optimizer.state_dict()}
   save_file_name = 'final_epoch_'+str(epoch)+'_iter_'+str(i)+'_acc_'+str(val_acc.item())
   torch.save(state, save_file_name)

def inference(device, img, checkpoint_file):
   model, _ , _, _= build_net_optimizer_schedular(cfg=None, device=device, mode='inference', checkpoint_file=checkpoint_file)
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
   if cfg.compute_dataset:
      compute_dataset_mean_std()
   else:
      train_net()