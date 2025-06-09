import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from glob import glob
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import cv2 as cv
from PIL import Image
import copy
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")


from model import FZNeT, Student, BN, Teachers
from std_resnet import wide_resnet50_2 as stwd
from resnet import wide_resnet50_2 as tcwd
from modules.dfs import DomainRelated_Feature_Selection
from requirements.update_modeler import updater 


from dataset import loading_dataset
from eval import evaluation_batch

import argparse


parser = argparse.ArgumentParser(description='FZA-Net ile anomali tespiti ve değerlendirme.')
parser.add_argument('--image_path', type=str, required=True,
                    help='Değerlendirilecek resmin dosya yolu.')

parser.add_argument('--image_size', type=int, default=224,
                    help='Modelin beklediği giriş resim boyutu (örn. 224).')

group = parser.add_mutually_exclusive_group()
group.add_argument('-q', '--quiet', action='store_true',
                    help='Sessiz çıktı modu. Sadece kritik bilgileri gösterir.')
group.add_argument('-v', '--verbose', action='store_true',
                    help='Detaylı çıktı modu. Ek bilgileri gösterir.')

args = parser.parse_args()



class Try(nn.Module):

    def __init__(self, args):
        super(Try, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        

        self.input_size = 224


        self.normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        Source_teacher, bn = tcwd(3)


        student = stwd(512)
        DFS = DomainRelated_Feature_Selection()

        Target_teacher = copy.deepcopy(Source_teacher)
        self.bn, self.DFS, self.student, self.Target_teacher = updater(bn, DFS, student, Target_teacher, device=self.device)


        bn.to(self.device)
        DFS.to(self.device)
        student.to(self.device)
        Target_teacher.to(self.device)
        Source_teacher.to(self.device) 


        self.model = FZNeT(
            dict(dataset_name="MVTec AD", image_size=self.input_size, setting="oc", batch_size=1,class_name="Instance"), # c dict'e image_size'ı args'tan al
            Source_teacher, Target_teacher, bn, student, DFS=DFS
        ).to(self.device)
        

        self.model.eval() 

    def linker(self, image_path):

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Resim dosyası bulunamadı: {image_path}")

        image = Image.open(image_path).convert('RGB')
        

        transform_x = transforms.Compose([
            transforms.Resize((224,224), transforms.InterpolationMode.LANCZOS), # Resize tuple olarak bekler
            transforms.ToTensor(),
            self.normalize_transform 
        ])
        
        processed_image = transform_x(image)

        return processed_image.unsqueeze(0).to(self.device)
    def forward(self, x):

        with torch.no_grad():
            _,_,stu_pred1_s = self.model(x)
        return stu_pred1_s[0][-1]

# --- Ana Program Akışı ---
if __name__ == "__main__":


    try_instance = Try(args=args)

    image_to_process_path = args.image_path

    if args.verbose:
        print(f"Hedef cihaz: {try_instance.device}")
        print(f"Model giriş boyutu: {try_instance.input_size}x{try_instance.input_size}")


    try:

        processed_image_tensor = try_instance.linker(image_to_process_path)
        
        if args.verbose:
            print(f"Resim başarıyla yüklendi ve işlendi. Tensor boyutu: {processed_image_tensor.shape}")

        if args.verbose:
            print("Modelin forward metodunu çağırıyor...")
        
        model_output = try_instance(processed_image_tensor) 
        
        if args.verbose:
            print("Model çıkışı alındı.")
            print(f"Model çıkış boyutu: {model_output.shape}")
            print(f"Modelin çıkışı: {model_output}")
        
        if args.quiet:
            print("Çıkarım tamamlandı.")
        elif not args.verbose:
            print(f"Çıkarım tamamlandı. Çıkış boyutu: {model_output.shape}")

    except FileNotFoundError as e:
        print(f"Hata: {e}. Lütfen resim dosya yolunu kontrol edin.")
    except Exception as e:
        print(f"Beklenmeyen bir hata oluştu: {e}")
