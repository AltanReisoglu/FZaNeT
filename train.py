import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from glob import glob
import os 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import cv2 as cv
from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
# from noise import Simplex_CLASS
import cv2
from model import FZNeT,Student,BN,Teachers
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from  std_resnet import wide_resnet50_2 as stwd
from resnet import wide_resnet50_2 as tcwd
from modules.dfs import DomainRelated_Feature_Selection
import warnings
import copy
warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")
from dataset import loading_dataset
from tqdm import tqdm
from eval import evaluation_batch
from accelerate import Accelerator
from requirements.update_modeler import updater
accelerator = Accelerator()

c=dict(dataset_name="MVTec AD",image_size=224,setting="oc",batch_size=1,epochs=5)
lr = {"lr_s": 1e-4, "lr_t": 1e-5}
def train_polyp(c):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    dataset_name = c["dataset_name"]


    # loading dataset
    train_dataloader, test_dataloader = loading_dataset(c)

    # model
    Source_teacher, bn = tcwd(3)

    student = stwd(512)
    DFS = DomainRelated_Feature_Selection()
    
    Target_teacher = copy.deepcopy(Source_teacher)
    
    bn,DFS,student,Target_teacher=updater(bn,DFS,student,Target_teacher)
    params = list(student.parameters()) + list(bn.parameters()) + list(DFS.parameters())
    optimizer = torch.optim.AdamW([{'params': student.parameters()},
                                   {'params': bn.parameters()},
                                   {'params': DFS.parameters()},
                                   {'params': Target_teacher.parameters(), 'lr': lr['lr_t']}],  # 5e-5
                                  lr=lr['lr_s'], betas=(0.9, 0.999), weight_decay=1e-5,  # 2e-3
                                  eps=1e-10, amsgrad=True)
    model = FZNeT(c, Source_teacher, Target_teacher, bn, student, DFS=DFS).to(device)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    total_iters = 5000
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(total_iters * 0.8)], gamma=0.2)
    it = 0
    best = 0.0


    for epoch in range(c["epochs"]):
        model.train_or_eval(type='train')
        loss_list = []

        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{c['epochs']}", leave=True)
        
        for i, sample in enumerate(progress_bar):
            img = sample[0].to(device)
            loss = model(img, mask=sample[2].to(device), max=False)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

            loss_list.append(loss.item())
            
            # tqdm bar'ında anlık loss'u göster
            progress_bar.set_postfix({"loss": loss.item()})

        # epoch sonu logları ve ağırlık kaydı
        print('Epoch [{}/{}] | Mean Loss: {:.4f}'.format(epoch+1, c["epochs"], np.mean(loss_list)))
        
        os.makedirs(f"weights/{epoch}", exist_ok=True)
        os.makedirs(f"weights/epoch_{epoch}_weights", exist_ok=True)
        
        torch.save(student.state_dict(), f"weights/epoch_{epoch}/epochs_{epoch}_student_trained.pth")
        torch.save(bn.state_dict(), f"weights/epoch_{epoch}/epochs_{epoch}_bn_trained.pth")
        torch.save(DFS.state_dict(), f"weights/epoch_{epoch}/epochs_{epoch}_dfs_trained.pth")
        torch.save(Target_teacher.state_dict(), f"weights/epoch_{epoch}/epochs_{epoch}_target_t_trained.pth")
        evaluation_batch(c,model,test_dataloader,device)
        return "okey"

train_polyp(c)