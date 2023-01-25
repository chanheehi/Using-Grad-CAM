import albumentations as A
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda
import numpy as np
import torch
from torchvision import datasets, transforms

def Get_train_transforms(img_size=256):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Cutout(num_holes=1, max_h_size=96, max_w_size=96, fill_value=230, p=0.5),   
        A.ShiftScaleRotate(
            shift_limit=0.1,    
            scale_limit=0.1,    
            rotate_limit=35,    
            p=0.8,              
        ),
        A.Normalize((0.485, 0.456, 0.456), (0.229, 0.224, 0.225)),
    ])

def Get_test_transforms(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize((0.485, 0.456, 0.456), (0.229, 0.224, 0.225)),
    ])