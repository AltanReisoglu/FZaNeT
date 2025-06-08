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

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            upsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,request=False
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if stride == 2:
            self.conv1 = UpConvAltan(inplanes, planes, bn=False,kernel_size=2,stride=stride)
        else:
            self.conv1 = ConvAltan(inplanes, planes,bn=False, kernel_size=3,stride=1,padding=1)

        if request==True:
            self.brute_attention=nn.Sequential(
                ASPP(planes,planes),
                CBAM(planes)   
            )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvAltan(planes, planes,bn=False, kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out
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
            downsample: Optional[nn.Module] = None,
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
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.mem = Mem(inplanes//2, width//2,stride)
        self.bn1 = norm_layer(width)
        
        self.conv2 = DepthwiseSeparableConv(width, width,bn=False,kernel_size=3, stride=stride,padding=dilation,groups=groups, dilation=dilation)
        
        
        if request==True:
            self.brute_attention=nn.Sequential(
                ASPP(inplanes,inplanes),
                CBAM(inplanes)   
            )
        else:
            self.brute_attention=nn.Identity()
            
        self.bn2 = norm_layer(width)
        self.conv3 = ConvAltan(width, planes * self.expansion,bn=False,kernel_size=1,stride=1,bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.bn4=norm_layer(inplanes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        x=self.brute_attention(x)
        out = self.mem(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        
        if self.downsample is not None:
            identity = self.downsample(x)
        
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
            block: Type[Union[BasicBlock, Bottleneck]],
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

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = ConvAltan(3, self.inplanes,bn=False, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0) 

  
    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):

            use_attention = (_ % 2 == 0)  # Her 3 bloktan birinde attention aktif
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, request=use_attention))

            if(blocks==3 or blocks==4):
                layers.append(Use_Def_att(self.inplanes))
            else:
                layers.append(AltanAttention(self.inplanes))
        return nn.Sequential(*layers)   

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feature_a = self.layer1(x)
        feature_b = self.layer2(feature_a)
        feature_c = self.layer3(feature_b)

        # x = self.avgpool(feature_d)
        # x = torch.flatten(x, 1)
        # res_logit = self.fc(x)

        return [feature_a, feature_b, feature_c]
class AttnBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            attention: bool = True,request=True
    ) -> None:
        super(AttnBasicBlock, self).__init__()
        self.attention = attention
        # print("Attention:", self.attention)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        # self.cbam = GLEAM(planes, 16)
        self.downsample = downsample
        self.stride = stride
        
        if request==True:
            self.brute_attention=nn.Sequential(
                ASPP(inplanes,inplanes),
                CBAM(inplanes),
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=True)
            )
        else:
            self.brute_attention=nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if self.attention:
        #    x = self.cbam(x)
        identity = x
        x=self.brute_attention(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AttnBottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            c,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            attention: bool = True,
            halve=2,request=True
    ) -> None:
        super(AttnBottleneck, self).__init__()
        self.attention = attention
        # print("Attention:",self.attention)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups  # 512
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.h = halve
        # self.k = [3, 5, 7, 13]
        # self.p = [1, 2, 3, 6]
        k = 7
        p = 3

        # self.conv1 = conv1x1(inplanes, width)
        # self.bn1 = norm_layer(width)
        # self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width // halve)
        self.conv3 = conv1x1(width, planes * self.expansion) 
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

        self.bn4 = norm_layer(width // 2)
        self.bn5 = norm_layer(width // 2)
        self.bn6 = norm_layer(width // 2)
        self.bn7 = norm_layer(width)
        self.conv3x3 = nn.Conv2d(inplanes // 2, width // 2, kernel_size=3, stride=stride, padding=1,
                                 bias=False)
        # self.Deconv3x3 = DepthwiseSeparableConv(inplanes // 2, width // 2, k=3, s=stride, p=1)
        self.conv3x3_ = nn.Conv2d(width // 2, width // 2, 3, 1, 1, bias=False)
        # self.Deconv3x3_ = DepthwiseSeparableConv(width // 2, width // 2, 3, 1, 1)
        self.conv7x7 = nn.Conv2d(inplanes // 2, width // 2, kernel_size=k, stride=stride, padding=p, bias=False)
        self.conv7x7_ = nn.Conv2d(width // 2, width // 2, k, 1, p, bias=False)

        if request==True:
            self.brute_attention=nn.Sequential(
                ASPP(inplanes//2,inplanes//2),
                CBAM(inplanes//2)   
            )
        else:
            self.brute_attention=nn.Identity()
    def get_same_kernel_bias(self):
        k1, b1 = fuse_bn(self.conv3x3, self.bn2)
        k2, b2 = fuse_bn(self.conv3x3_, self.bn6)

        return k1, b1, k2, b2

    def merge_kernel(self):
        k1, b1, k2, b2 = self.get_same_kernel_bias()
        self.conv7x7 = nn.Conv2d(self.conv3x3.in_channels, self.conv3x3.out_channels, self.conv3x3.kernel_size,
                                 self.conv3x3.stride, self.conv3x3.padding, self.conv3x3.dilation,
                                 self.conv3x3.groups)
        self.conv7x7_ = nn.Conv2d(self.conv3x3_.in_channels, self.conv3x3_.out_channels, self.conv3x3_.kernel_size,
                                  self.conv3x3_.stride, self.conv3x3_.padding, self.conv3x3_.dilation,
                                  self.conv3x3_.groups)
        self.conv7x7.weight.data = k1
        self.conv7x7.bias.data = b1
        self.conv7x7_.weight.data = k2
        self.conv7x7_.bias.data = b2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.h == 1:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

        else:

            C = x.shape[1]
            x_ = torch.split(x, [C // 2, C // 2], dim=1)

            def process_branch(branch, conv1, bn1, conv2, bn2, relu):
                branch=self.brute_attention(branch)
                out = conv1(branch)
                out = bn1(out)
                out = relu(out)
                out = conv2(out)
                out = bn2(out)
                return out

            out1 = process_branch(x_[0], self.conv3x3, self.bn2, self.conv3x3_, self.bn5, self.relu)
            out2 = process_branch(x_[-1], self.conv7x7, self.bn4, self.conv7x7_, self.bn6, self.relu)



            out = torch.cat([out1, out2], dim=1)
            # out = self.bn7(out)
            # out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class BN_layer(nn.Module):
    def __init__(self,
                 c,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: int,
                 groups: int = 1,
                 width_per_group: int = 64,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 ):
        super(BN_layer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 256 * block.expansion
        self.dilation = 1
        self.h = 2
        self.bn_layer = nn.Sequential(self._make_layer(c, block, 512, layers, stride=2),)
        self.conv1 = conv3x3(64 * block.expansion, 128 * block.expansion, 2)
        self.bn1 = norm_layer(128 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn2 = norm_layer(256 * block.expansion)
        self.conv3 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn3 = norm_layer(256 * block.expansion)

        self.conv4 = conv1x1(1024 * block.expansion, 512 * block.expansion, 1)
        self.bn4 = norm_layer(512 * block.expansion)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif hasattr(m, "merge_kernel") and self.h == 2:
                m.merge_kernel()

    def _make_layer(self, c, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes * 3, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(c, self.inplanes * 3, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, halve=self.h))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(c, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, halve=self.h))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # print(x[0].shape)
        l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x[0]))))))
        l2 = self.relu(self.bn3(self.conv3(x[1])))
        feature = torch.cat([l1, l2, x[2]], 1)
        output = self.bn_layer(feature)  # 16*2048*8*8
        # print(output.shape)

        return output.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

def _resnet(
        block: Type[Union[BasicBlock, Bottleneck]],
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
                   pretrained, progress, **kwargs), BN_layer(c, AttnBottleneck, 3, **kwargs)

if __name__=="__main__":
    
    model = wide_resnet50_2(3)[0].to("cuda")
    model2 = wide_resnet50_2(3)[1].to("cuda")
    tensor=torch.rand(2,3,224,224).to("cuda")
    
    print(model2(model(tensor)).shape)


