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
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")

class MultiMVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good' or defect_type == 'ok':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path

def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms
    
class MVTecDataset(Dataset):
    def __init__(self, c, is_train=True, dataset='mvtec'):
        self.dataset_path = "datasets"
        self.class_name = c["class_name"]
        self.is_train = is_train
        # self.is_vis = c.is_vis
        self.input_size = (c["image_size"], c["image_size"])
        self.aug = False
        phase = 'train' if self.is_train else 'test'
        self.img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        self.gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')
        # load dataset
        self.x, self.y, self.mask, _ = self.load_dataset()
        # set transforms
        if is_train:
                self.transform_x = transforms.Compose([
                    transforms.Resize(self.input_size, transforms.InterpolationMode.LANCZOS),
                    transforms.RandomPerspective(distortion_scale=0.5, p=1.0),
                    transforms.ToTensor()])
        # test:
        else:
            self.transform_x = transforms.Compose([
                transforms.Resize(self.input_size, transforms.InterpolationMode.LANCZOS),
                transforms.ToTensor()])
        # mask
        self.transform_mask = transforms.Compose([
            transforms.Resize(self.input_size, transforms.InterpolationMode.NEAREST),
            transforms.RandomPerspective(distortion_scale=0.5, p=1.0),
            transforms.ToTensor()])

        self.normalize = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        
        x = Image.open(x)

        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            x = np.expand_dims(np.array(x), axis=2)
            x = np.concatenate([x, x, x], axis=2)

            x = Image.fromarray(x.astype('uint8')).convert('RGB')
        #
        x = self.normalize(self.transform_x(x))
        #
        if y == 0:
            mask = torch.zeros([1, *self.input_size])
        else:
            mask = cv.imread(mask, cv.IMREAD_GRAYSCALE)  # grayscale olarak oku (isteğe bağlı)
            mask = Image.fromarray(mask)                 # NumPy → PIL
            mask = self.transform_mask(mask)             # Artık transform uygulanabilir
        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset(self):

        img_tot_paths = list()
        gt_tot_paths = list()
        tot_labels = list()
        tot_types = list()

        defect_types = os.listdir(self.img_dir)
        
        for defect_type in defect_types:
            # if self.is_vis and defect_type == "good":
                # continue
            if defect_type == 'good':
                
                img_paths = glob.glob(os.path.join(self.img_dir, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([None] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))

            else:
                img_paths = glob.glob(os.path.join(self.img_dir, defect_type) + "/*")
                gt_paths = glob.glob(os.path.join(self.gt_dir, defect_type) + "/*")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
        
        assert len(img_tot_paths) == len(tot_labels), "Something wrong with test and ground truth pair!"
       
        return img_tot_paths, tot_labels, gt_tot_paths, tot_types
mvtec_list = ['capsule', 'pill','transistor']


def loading_dataset(c):
    train_dataloader, test_dataloader = None, None


    if c["dataset_name"] == 'MVTec AD' and c["setting"] == 'oc':

        train_data_list = []
        test_data_list = []
        # train_path = './mvtec/' + c._class_ + '/train'
        # data_transform, gt_transform = get_data_transforms(c.image_size, c.image_size)
        # train_data = NoiseMVTecDataset(root=train_path, transform=data_transform)
        # train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=c.batch_size, shuffle=True)
        for _,class_name in enumerate(mvtec_list):
            c["class_name"]=class_name
            train_data = MVTecDataset(c, is_train=True)
            test_data = MVTecDataset(c, is_train=False)
            train_data_list.append(train_data)
            test_data_list.append(test_data)
        main_train_dataset=ConcatDataset(train_data_list)
        main_test_dataset=ConcatDataset(test_data_list)

        train_dataloader = torch.utils.data.DataLoader(main_train_dataset, batch_size=c["batch_size"], shuffle=True,
                                                        pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(main_test_dataset, batch_size=1, shuffle=False, pin_memory=True)

        return train_dataloader,test_dataloader

    elif c.dataset_name in ['MVTec AD', 'BTAD', 'MVTec 3D-AD', "VisA"] and c.setting == 'mc':
        data_transform, gt_transform = get_data_transforms(224,224)
        train_data_list = []
        test_data_list = []

        lr = {"lr_s": 1e-4, "lr_t": 1e-5}

        if dataset_name == 'MVTec AD':
            dataset_name = 'mvtec'
            class_list = mvtec_list
    


        for i, item in enumerate(class_list):
            train_path = '../data/{}/'.format(dataset_name) + item + '/train'
            test_path = '../data/{}/'.format(dataset_name) + item

            train_data = ImageFolder(root=train_path, transform=data_transform)
            train_data.classes = item
            train_data.class_to_idx = {item: i}
            train_data.samples = [(sample[0], i) for sample in train_data.samples]

            test_data = MultiMVTecDataset(root=test_path, transform=data_transform,
                                          gt_transform=gt_transform, phase="test")
            train_data_list.append(train_data)
            test_data_list.append(test_data)

        train_data = ConcatDataset(train_data_list)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=c.batch_size, shuffle=True,
                                                       num_workers=1, drop_last=True)
        test_dataloader_list = [
            torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
            for test_data in test_data_list]

        return train_dataloader, test_dataloader_list, class_list, lr



if __name__=="__main__":
    
    c=dict(dataset_name="MVTec AD",image_size=224,setting="oc",batch_size=1,epoch=5)
    train,test=loading_dataset(c)
    for a, (x, mask, label) in enumerate(train):
        print(f"Batch {a}")
        print("x NaN:", torch.isnan(x).any(), "Inf:", torch.isinf(x).any(), "AllZero:", torch.all(x == 0))
        print("mask NaN:", torch.isnan(mask).any(), "Inf:", torch.isinf(mask).any(), "AllZero:", torch.all(mask == 0))
        print("label:", label)
        print("xdxd")

