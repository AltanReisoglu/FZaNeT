import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import wide_resnet50_2,resnet50,ResNet50_Weights
from typing import Type, Any, Callable, Union, List, Optional
from modules.layers import ConvAltan,UpConvAltan,SpatialAttention,ChannelAttentionAdaptive,ASPP,CBAM,SelfAttention
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import gc
from torchsummary import summary

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class Mem(nn.Module):
    
    @staticmethod
    def fuse_bn(conv, bn):
        # Ağırlık ve bias
        W_conv = conv.weight.clone()
        b_conv = conv.bias if conv.bias is not None else torch.zeros(W_conv.size(0), device=W_conv.device)

        # BN parametreleri
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        device = W_conv.device
        gamma = gamma.to(device)
        beta = beta.to(device)
        mean = mean.to(device)
        var = var.to(device)

        # Hesaplama
        std = torch.sqrt(var + eps)
        W_fused = W_conv * (gamma / std).reshape([-1, 1, 1, 1])
        b_fused = (b_conv - mean) / std * gamma + beta

        return W_fused, b_fused
    """Some Information about Mem"""
    def __init__(self,half_in,istenc,stride=1):
        super(Mem, self).__init__()

        norm_layer=nn.BatchNorm2d

        k = 7
        p = 3

        self.conv3x3 = nn.Conv2d( half_in, istenc, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3x3_ = nn.Conv2d( istenc,istenc, kernel_size=3, stride=1, padding=1, bias=False)
        # self.Deconv3x3_ = DepthwiseSeparableConv(width // 2, width // 2, 3, 1, 1)
        self.conv7x7 = nn.Conv2d( half_in,  istenc, kernel_size=k, stride=1, padding=p, bias=False)
        self.conv7x7_ = nn.Conv2d(istenc,istenc, kernel_size=k, stride=1, padding=p, bias=False)
        self.bn_f=norm_layer(istenc)
        self.bn_s=norm_layer(istenc)
        self.bn_f_p=norm_layer(istenc)
        self.bn_s_p=norm_layer(istenc)
        self.relu=nn.ReLU(inplace=True)
        
    def pre_forward(self):
        k1, b1 = Mem.fuse_bn(self.conv3x3, self.bn_f)
        k2, b2 = Mem.fuse_bn(self.conv3x3_, self.bn_s)

        return k1, b1, k2, b2
    
    def pre_post_forward(self):
        k1,b1,k2,b2=self.pre_forward()
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

    def forward(self, x):
        C = x.shape[1]
        x_ = torch.chunk(x,2,dim=1)
        self.pre_post_forward()
        def process_branch(branch, conv1, bn1, conv2, bn2, relu):
                out = conv1(branch)
                out = bn1(out)
                out = relu(out)
                out = conv2(out)
                out = bn2(out)
                return out

        out1 = process_branch(x_[0], self.conv3x3, self.bn_f, self.conv3x3_, self.bn_f_p, self.relu)
        out2 = process_branch(x_[-1], self.conv7x7, self.bn_s, self.conv7x7_, self.bn_s_p, self.relu)
        out = torch.cat([out1, out2], dim=1)
          
        
        return out
    
if __name__ =="__main__":
    model = Mem(32,512).to("cuda")
    tensor=torch.rand(1,64,56,56).to("cuda")
    print(model(tensor).shape)

