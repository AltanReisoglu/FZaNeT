import torch
import torch.nn as nn
from modules.loss import losses,structure_loss
import numpy as np
from torch.nn import functional as F
from resnet import wide_resnet50_2 as wd
from std_resnet import wide_resnet50_2 as wd2
from modules.dfs import DomainRelated_Feature_Selection,domain_related_feature_selection

class EarlyStopping:
    def __init__(self, patience=3, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.counter = 0
        self.best_pro = None
        self.early_stop = False
        self.pro_min = 0.0

    def __call__(self, pro):
        current_pro = pro

        if self.best_pro is None:
            self.best_pro = current_pro
        elif current_pro < self.best_pro + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_pro = current_pro
            self.save_checkpoint(current_pro)
            self.counter = 0

    def save_checkpoint(self, pro):

        if self.verbose:
            print(f'PRO increases ({self.pro_min:.1f} --> {pro:.1f}). Saving model...')
        self.pro_min = pro


class FZNeT(nn.Module):
    def __init__(self, c, Source_teacher, Target_teacher, bottleneck, student, DFS=None):
        super().__init__()
        self._class_ = c["class_name"]
        self.T = 0.1
        self.n = 1 if Target_teacher is None else 2
        self.t = Teachers(Source_teacher=Source_teacher, Target_teacher=Target_teacher)
        self.bn = BN(bottleneck)
        self.s = Student(student=student)
        self.dfs = DFS

    def train_or_eval(self, type='train'):
        self.type = type
        self.t.train_eval(type)
        self.bn.train_eval(type)
        self.s.train_eval(type)

        return self

    def feature_selection(self, b, a, max):
        if self._class_ in ['transistor']:
            return a

        if self.dfs is not None:
            selected_features = self.dfs(a, b, learnable=True, conv=False, max=max)
        else:
            from modules.dfs import domain_related_feature_selection
            selected_features = domain_related_feature_selection(a, b, max=max)
        return selected_features

    def loss_computation(self, b, a, pred_list, margin=1, mask=None, stop_gradient=False):
        loss = losses(b, a, self.T, margin, mask=None, stop_gradient=stop_gradient) + \
               structure_loss(pred_list[0][0], mask) + structure_loss(pred_list[0][-1], mask)
                 
        return loss
                               
    def forward(self, x, max=True, mask=None, stop_gradient=False):
        Sou_Tar_features, bnins = self.t(x)
        bnsout = self.bn(bnins)
        stu_features, stu_pred = self.s(bnsout, [bnins[-1], bnins[-2]])
        ##stupred [torch.tensor]
        stu_features = [d.chunk(dim=0, chunks=2) for d in stu_features]
        stu_features = [stu_features[2][0], stu_features[1][0], stu_features[0][0],
                        stu_features[2][1], stu_features[1][1], stu_features[0][1]]
        
        stu_pred1_f = F.interpolate(stu_pred[0], scale_factor=4, mode='bilinear')
        stu_pred1_s = stu_pred1_f.chunk(dim=0, chunks=2)
        
        if self.type == 'train':
            stu_features_ = self.feature_selection(Sou_Tar_features,stu_features, max)
            
            loss = self.loss_computation(Sou_Tar_features, stu_features_,[stu_pred1_s], mask=mask, stop_gradient=stop_gradient)

            return loss
        else:
            return Sou_Tar_features, stu_features,[stu_pred1_s]


class Teachers(nn.Module):
    def __init__(self, Source_teacher, Target_teacher):
        super().__init__()
        self.t_s = Source_teacher
        self.t_t = Target_teacher

    def train_eval(self, type='train'):
        self.type = type
        self.t_s.eval()
        if self.t_t is not None:
            if type == "train":
                self.t_t.train()
            else:
                self.t_t.eval()

        return self

    def forward(self, x):
        with torch.no_grad():
            Sou_features = self.t_s(x)

        if self.t_t is None:
            return Sou_features
        else:
            Tar_features = self.t_t(x)
            bnins = [torch.cat([a, b], dim=0) for a, b in zip(Tar_features, Sou_features)]  # 512, 1024, 2048
            return Sou_features + Tar_features, bnins


class BN(nn.Module):
    def __init__(self, bottleneck):
        super().__init__()
        self.bn = bottleneck

    def train_eval(self, type='train'):
        if type == 'train':
            self.bn.train()
        else:
            self.bn.eval()

        return self

    def forward(self, x):
        bns = self.bn(x)

        return bns


class Student(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.s1 = student

    def train_eval(self, type='train'):
        if type == 'train':
            self.s1.train()
        else:
            self.s1.eval()

        return self

    def forward(self, bn_outs, skips=None):
        de_features = self.s1(bn_outs,skips)

        return de_features

if __name__=="__main__":
    c=dict(dataset_name="MVTec AD",image_size=224,setting="oc",batch_size=1,epochs=5,class_name="pill")
    source,bn_1=wd(3)
    target,bn_2=wd(3)
    student=wd2(512)
    tenspr=torch.rand(1,3,224,224).to("cuda")
    model=FZNeT(c,source,target,bn_1,student).to("cuda")
    sonuc=model(tenspr)
    
    print(sonuc)


    