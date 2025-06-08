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

# Model importları - Bu modüllerin projede mevcut olduğunu varsayıyoruz
from model import FZNeT, Student, BN, Teachers
from std_resnet import wide_resnet50_2 as stwd
from resnet import wide_resnet50_2 as tcwd
from modules.dfs import DomainRelated_Feature_Selection
from requirements.update_modeler import updater # updater fonksiyonunuz

# Veri kümesi ve değerlendirme importları
from dataset import loading_dataset
from eval import evaluation_batch

import argparse

# --- Argparse kısmını burada, sadece ilgili argümanlarla tanımlıyoruz ---
parser = argparse.ArgumentParser(description='FZA-Net ile anomali tespiti ve değerlendirme.')
parser.add_argument('--image_path', type=str, required=True,
                    help='Değerlendirilecek resmin dosya yolu.')

parser.add_argument('--image_size', type=int, default=224,
                    help='Modelin beklediği giriş resim boyutu (örn. 224).')

# Mutually exclusive group: Sadece birinin seçilebileceği argümanlar
group = parser.add_mutually_exclusive_group()
group.add_argument('-q', '--quiet', action='store_true',
                    help='Sessiz çıktı modu. Sadece kritik bilgileri gösterir.')
group.add_argument('-v', '--verbose', action='store_true',
                    help='Detaylı çıktı modu. Ek bilgileri gösterir.')

args = parser.parse_args()
# --- Argparse kısmı burada sona eriyor ---


class Try(nn.Module):
    # args nesnesini __init__'e parametre olarak geçiriyoruz
    def __init__(self, args):
        super(Try, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # argparse'dan gelen image_size'ı kullanıyoruz
        self.input_size = 224

        # Normalize transform'ı tanımlıyoruz. Genellikle ImageNet ön-eğitimli modeller için standart değerlerdir.
        self.normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Model bileşenlerini başlatma
        # tcwd fonksiyonunun iki değer döndürdüğünü varsayıyoruz: Source_teacher ve bn
        # Bu 3 parametresinin ne anlama geldiğini kendi model tanımınıza göre kontrol edin (genellikle output/input channels)
        Source_teacher, bn = tcwd(3)

        # Bu 512 parametresinin ne anlama geldiğini kendi model tanımınıza göre kontrol edin
        student = stwd(512)
        DFS = DomainRelated_Feature_Selection()

        Target_teacher = copy.deepcopy(Source_teacher)


        
        # Bu kısımda hala hata alıyorsanız (RuntimeError: size mismatch vb.),
        # model mimarilerinizin (stwd, tcwd, DomainRelated_Feature_Selection sınıflarınızın)
        # .pth dosyalarını kaydeden modellerle tam olarak eşleştiğinden emin olun.
        # Özellikle Conv2d katmanlarının kernel_size ve bias parametrelerini kontrol edin.
        self.bn, self.DFS, self.student, self.Target_teacher = updater(bn, DFS, student, Target_teacher, device=self.device)

        # Modelleri cihaza taşıma
        bn.to(self.device)
        DFS.to(self.device)
        student.to(self.device)
        Target_teacher.to(self.device)
        Source_teacher.to(self.device) # Source_teacher da muhtemelen kullanılacak, onu da taşıyalım

        # FZNeT modelini oluşturma ve cihaza taşıma
        self.model = FZNeT(
            dict(dataset_name="MVTec AD", image_size=self.input_size, setting="oc", batch_size=1,class_name="Instance"), # c dict'e image_size'ı args'tan al
            Source_teacher, Target_teacher, bn, student, DFS=DFS
        ).to(self.device)
        
        # Grad norm klibini eğitim aşamasında kullanırsınız, burada yorum satırı yaptım.
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Değerlendirme modunu ayarlama
        self.model.eval() # Yükledikten sonra genellikle değerlendirme moduna alınır

    def linker(self, image_path):

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Resim dosyası bulunamadı: {image_path}")

        image = Image.open(image_path).convert('RGB') # Resmi RGB formatında aç
        
        # __init__ metodunda tanımlanan transformları kullanarak resmi işle
        # InterpolationMode, torchvision.transforms'tan import edilmeli
        transform_x = transforms.Compose([
            transforms.Resize((224,224), transforms.InterpolationMode.LANCZOS), # Resize tuple olarak bekler
            transforms.ToTensor(),
            self.normalize_transform # Normalize transform'ı kullan
        ])
        
        processed_image = transform_x(image)
        # Modeller genellikle batch input bekler, bu yüzden bir batch boyutu ekleyelim
        return processed_image.unsqueeze(0).to(self.device) # Modeli beslemek için batch boyutu ekle ve cihaza taşı

    def forward(self, x):

        with torch.no_grad(): # Çıkarım (inference) yaparken gradyan hesaplamayı devre dışı bırakır
            _,_,stu_pred1_s = self.model(x)
        return stu_pred1_s[0][-1]

# --- Ana Program Akışı ---
if __name__ == "__main__":
    # `args` nesnesi yukarıda `parser.parse_args()` ile zaten oluşturuldu.
    
    # Try sınıfından bir örnek oluşturun, args nesnesini ona geçirerek
    try_instance = Try(args=args)

    # Argparse'tan gelen resim yolunu kullanıyoruz
    image_to_process_path = args.image_path

    if args.verbose:
        print(f"Hedef cihaz: {try_instance.device}")
        print(f"Model giriş boyutu: {try_instance.input_size}x{try_instance.input_size}")


    try:
        # linker metodunu kullanarak resmi yükle ve işle
        processed_image_tensor = try_instance.linker(image_to_process_path)
        
        if args.verbose:
            print(f"Resim başarıyla yüklendi ve işlendi. Tensor boyutu: {processed_image_tensor.shape}")

        # İşlenmiş tensörü forward metoduna (yani FZNeT modeline) besle
        if args.verbose:
            print("Modelin forward metodunu çağırıyor...")
        
        model_output = try_instance(processed_image_tensor) # Bu, Try'ın forward'ını çağırır, o da içindeki self.model'i çağırır.
        
        if args.verbose:
            print("Model çıkışı alındı.")
            print(f"Model çıkış boyutu: {model_output.shape}")
            print(f"Modelin çıkışı: {model_output}")
            # print(f"Model çıkışı (ilk birkaç değer): {model_output.flatten()[:5]}") # İsteğe bağlı
        
        if args.quiet:
            print("Çıkarım tamamlandı.")
        elif not args.verbose: # Ne quiet ne de verbose ise varsayılan özet
            print(f"Çıkarım tamamlandı. Çıkış boyutu: {model_output.shape}")

    except FileNotFoundError as e:
        print(f"Hata: {e}. Lütfen resim dosya yolunu kontrol edin.")
    except Exception as e:
        print(f"Beklenmeyen bir hata oluştu: {e}")