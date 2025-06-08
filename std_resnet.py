import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import wide_resnet50_2,resnet50,ResNet50_Weights
from typing import Type, Any, Callable, Union, List, Optional
from modules.layers import ConvAltan,UpConvAltan,SpatialAttention,ChannelAttentionAdaptive,ASPP,CBAM,SelfAttention,DepthwiseSeparableConv
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import gc
from torchsummary import summary

from mem import Mem
from modules.deformable_attention import Use_Def_att
torch.cuda.empty_cache()
gc.collect()
def fuse_bn(conv, bn):


    # Orijinal ağırlıkları al
    W_conv = conv.weight.clone()
    if conv.bias is not None:
        b_conv = conv.bias.clone()
    else:
        b_conv = torch.zeros(conv.out_channels)

    # BN parametreleri
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    # Yeni ağırlıklar ve bias
    std = torch.sqrt(var + eps)
    W_fused = W_conv * (gamma / std).reshape(-1, 1, 1, 1)
    b_fused = (b_conv - mean) / std * gamma + beta

    return W_fused,b_fused


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):


    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            upsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,request=False,halve=2
    ) -> None:
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.h = halve
        width = int(planes * (base_width / 64.)) * groups
        self.mem = self.conv1 = conv1x1(inplanes, width)

        self.bn1 = norm_layer(width)
        if stride==2:
            self.conv2=UpConvAltan(width,width,bn=False,kernel_size=2,stride=stride,groups=1,bias=False,dilation=dilation)
        else:
            self.conv2 = DepthwiseSeparableConv(width, width,bn=False,kernel_size=3,stride=stride,padding=dilation,groups=groups,dilation=dilation)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        
        self.bn2 = norm_layer(width)
        
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        
        if request==True:
            self.brute_attention=nn.Sequential(
                ASPP(inplanes,inplanes),
                CBAM(inplanes),
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=True)
                
            )
        else:
            self.brute_attention=nn.Identity()
            

        self.relu=nn.ReLU()

        self.upsample = upsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        x=self.brute_attention(x)
        out = self.mem(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out=self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class AltanAttention(nn.Module):

    def __init__(self,n_embd:int,n_head:int=1,request:bool=False):
        super().__init__()
        
        self.attention=SelfAttention(n_embd,n_head,request)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        B,C,H,W=x.shape
        residue=x
        x=x.view((B,C,-1))
        x=torch.transpose(x,-1,-2)
        x=self.attention(x)
        x=torch.transpose(x,-1,-2)
        x=x.view((B,C,H,W))
        x=residue+x
        return x

class BaseModel(nn.Module):
    def __init__(
            self,
            block: Type[Union[Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_classes = num_classes

        self.inplanes = 512*block.expansion
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        """self.conv1 = ConvAltan(3, self.inplanes,bn=False, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)"""
        self.layer1 = self._make_layer(block, 256, layers[0],stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0) 

        self.conv1x1_1 = nn.Conv2d(256, 1, 1, 1, 0, bias=False)
        #nn.Conv2d(256, 1, 1, 1, 0, bias=False)
    def Basic(self, intInput, intOutput):
        return torch.nn.Sequential(
            torch.nn.Conv2d(intInput, intOutput, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(intOutput),
            torch.nn.ReLU(inplace=True),
            # torch.nn.Conv2d(intOutput, intOutput, kernel_size=3, padding=1),
            # torch.nn.BatchNorm2d(intOutput),
            # torch.nn.ReLU(inplace=True)
        )

    def Upsample(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )


    def Generator(self, intInput, intOutput, nc):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(nc),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(nc),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            # torch.nn.BatchNorm2d(intOutput),
            # torch.nn.ReLU(inplace=True),
        )


  
    def _make_layer(self, block: Type[Union[Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                UpConvAltan(self.inplanes, planes * block.expansion,bn=False,kernel_size=2,stride=stride,bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):

            use_attention = (_ % 2 == 0)  # Her 3 bloktan birinde attention aktif
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, request=use_attention))
            if(blocks%6==0 or blocks%4==0):
                layers.append(Use_Def_att(self.inplanes))
            else:
                layers.append(AltanAttention(self.inplanes))
        return nn.Sequential(*layers)   

    def forward(self,x:torch.Tensor,skip=None)->torch.Tensor:

        feature_a = self.layer1(x)

        if skip is not None:
            feature_b = self.layer2(feature_a+skip[0])
            feature_c = self.layer3(feature_b+skip[1])
        else:
            feature_b = self.layer2(feature_a)
            feature_c = self.layer3(feature_b)

        feature_c_l=self.conv1x1_1(feature_c)
        


        # x = self.avgpool(feature_d)
        # x = torch.flatten(x, 1)
        # res_logit = self.fc(x)

        return [feature_a, feature_b, feature_c],[feature_c_l]
    
        de_f3 = self.moduleDeconv3(feature_c)
        up2 = self.up2(de_f3)

        de_f2 = self.moduleDeconv2(up2)
        up1 = self.up1(de_f2)

        de_f1 = self.moduleDeconv1(up1)
        out = self.generator(de_f1)

        return [feature_c, feature_b, feature_a], out

def _resnet(
        block: Type[Union[Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> BaseModel:
    model = BaseModel(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
                                              progress=progress)
        # for k,v in list(state_dict.items()):
        #    if 'layer4' in k or 'fc' in k:
        #        state_dict.pop(k)
        model.load_state_dict(state_dict)
    # else:
    #     model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

def wide_resnet50_2(c, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> BaseModel:

    kwargs['width_per_group'] = 64 * 2
    return _resnet(Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)

if __name__=="__main__":
    model = wide_resnet50_2(512).to("cuda:1")
    tensor=torch.rand(2,2048,7,7).to("cuda:1")
    for i in range(len(model(tensor)[1])):
        
        print(model(tensor)[1][i].shape)

