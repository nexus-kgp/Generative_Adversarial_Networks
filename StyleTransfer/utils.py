import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms

def get_loader(config):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""
    
    transform = transforms.Compose([
                    transforms.Scale(config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    svhn = datasets.SVHN(root=config.svhn_path, download=True, transform=transform)
    mnist = datasets.MNIST(root=config.mnist_path, download=True, transform=transform)

    svhn_loader = torch.utils.data.DataLoader(dataset=svhn,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.num_workers)

    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)
    return svhn_loader, mnist_loader

def merge_images(sources, targets, batch_size=64):
      _, _, h, w = sources.shape
      row = int(np.sqrt(batch_size))
      merged = np.zeros([3, row*h, row*w*2])

      for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row
        j = idx % row

        merged[:, i*h : (i+1)*h, (j*2)*h : (j*2+1)*h] = s
        merged[:, i*h : (i+1)*h, (j*2+1)*h : (j*2+2)*h] = t
      return merged.transpose(1, 2, 0)

def deconv_layer(c_in, c_out, k_size, stride=2, pad=1, batch_norm=True):
  layers = []
  layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
  
  if batch_norm:
    layers.append(nn.BatchNorm2d(c_out))
  
  return nn.Sequential(*layers)

def conv_layer(c_in, c_out, k_size, stride=2, pad=1, batch_norm=True):
  layers = []
  layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))

  if batch_norm:
    layers.append(nn.BatchNorm2d(c_out))

  return nn.Sequential(*layers)

def residual_layer(x, c_in, c_out, k_size, stride=2, pad=1, batch_norm=True):
  #TODO
  pass