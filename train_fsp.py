import torch
import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import random_split, dataloader
from src.model import *

def main():
    for i in range(20):
        np.random.seed(i)
        torch.manual_seed(i)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data_dir = './data/cifar10'
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean = [0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        cifar10_train = datasets.CIFAR10(root=data_dir,
                                        download=True,
                                        train=True,
                                        transform=transform)
        cifar10_test = datasets.CIFAR10(root=data_dir,
                                        download=True,
                                        train=False,
                                        transform=transform)
        n_samples = len(cifar10_train)
        n_train = int(n_samples * 0.8)
        n_val = n_samples - n_train
        cifar10_train, cifar10_val = random_split(cifar10_train, [n_train, n_val])
        
        train_dataloader = dataloader(cifar10_train,
                                      batch_size=128,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=8)
        val_dataloader = dataloader(cifar10_val,
                                    batch_size=128,
                                    shuffle=False)
        test_dataloader = dataloader(cifar10_test,
                                    batch_size=128,
                                    shuffle=False)
        
        teacher = resnet50().to(device)
        student = resnet18().to(device)
        
        
        
if __name__ == '__main__':
    main()
    