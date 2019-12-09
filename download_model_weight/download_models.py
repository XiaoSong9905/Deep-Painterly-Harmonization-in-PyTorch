import torch
from os import path
from sys import version_info
from collections import OrderedDict
from torch.utils.model_zoo import load_url

# Code adapted from https://github.com/ProGamerGov/neural-style-pt 

# Download the VGG-19 model and fix the layer names
print("Downloading the VGG-19 model")
sd = load_url("https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg19-d01eb7cb.pth")
map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
sd = OrderedDict([(map[k] if k in map else k,v) for k,v in sd.items()])
torch.save(sd, path.join("download_model_weight", "vgg19-d01eb7cb.pth"))

# Download the VGG-16 model and fix the layer names
#print("Downloading the VGG-16 model")
#sd = load_url("https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth")
#map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
#sd = OrderedDict([(map[k] if k in map else k,v) for k,v in sd.items()])
#torch.save(sd, path.join("download_model_weight", "vgg16-00b39a1b.pth"))
