
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ConvAltan(nn.Module):
    
    def __init__(self,in_channels,out_channels,bn=False,**kwargs):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,**kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn
    def get_conv(self):
        return self.conv1
    def forward(self, x):
        if(self.use_bn_act):
            x=self.conv1(x)
            x=self.bn(x)
            x=self.leaky(x)
        else:
            x=self.conv1(x)
        return x
    
class UpConvAltan(nn.Module):
    
    def __init__(self,in_channels,out_channels,bn=False,**kwargs):
        super().__init__()
        self.conv1=nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,**kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn
    def forward(self, x):
        if(self.use_bn_act):
            x=self.conv1(x)
            x=self.bn(x)
            x=self.leaky(x)
        else:
            x=self.conv1(x)
        return x
    


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels,bn, kernel_size=3, stride=1, padding=1,groups=16,dilation=1,*kwargs):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = ConvAltan(
            in_channels, in_channels,bn=False ,kernel_size=kernel_size,
            stride=stride, padding=padding,bias=True,groups=groups,dilation=dilation,*kwargs
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels ,kernel_size=1, bias=False
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

    

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = ConvAltan(in_channels, out_channels,False, kernel_size=1, padding=0, dilation=1)
        self.atrous_block6 = ConvAltan(in_channels, out_channels, False,kernel_size=3, padding=6, dilation=6)
        self.atrous_block12 = ConvAltan(in_channels, out_channels, False,kernel_size=3, padding=12, dilation=12)
        self.atrous_block18 = ConvAltan(in_channels, out_channels, False,kernel_size=3, padding=18, dilation=18)
        self.conv_1x1_output = ConvAltan(out_channels * 4, out_channels,True, kernel_size=1,bias=False)

    def forward(self, x):
        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv_1x1_output(x_cat)
    

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ChannelAttentionAdaptive(nn.Module):
    #Feature map'in kanallar arası önem derecesini öğrenmeye çalışır. 
    # Hangi feature channel daha bilgilendirici ise onun ağırlığını artırır.
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionAdaptive, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.max_pool = nn.AdaptiveMaxPool2d(1)  

        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        avg_out = self.shared_mlp(self.avg_pool(x))  
        max_out = self.shared_mlp(self.max_pool(x)) 

        attention = self.sigmoid(avg_out + max_out)  

        return x * attention  
    
class SpatialAttention(nn.Module):
    #Feature map üzerinde hangi konumsal bölgelerin daha önemli olduğunu öğrenir.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv=nn.Conv2d(2,1,kernel_size=7,padding=7//2)
    def forward(self,x):
        avg_out=torch.mean(x,dim=1,keepdim=True)
        max_out,_=torch.max(x,dim=1,keepdim=True)
        concat=torch.cat([avg_out,max_out],dim=1)
        out=self.conv(concat)
        out=torch.sigmoid(out)
        return out*x
    
class CBAM(nn.Module):

    def __init__(self,in_channels):
        super(CBAM, self).__init__()
        self.chanel_attention=ChannelAttentionAdaptive(in_channels)
        self.spatial_attention=SpatialAttention()

    def forward(self, x):
        x=self.chanel_attention(x)
        x=self.spatial_attention(x)
        return x
    

    
class AdditiveAttention(nn.Module):
    #Özellikle RNN tabanlı encoder-decoder mimarilerde, sıralı veri üzerinde ilgili zaman adımlarını seçmek için kullanılır.
    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        self.W_q = nn.Linear(query_dim, hidden_dim)  # W₁
        self.W_k = nn.Linear(key_dim, hidden_dim)    # W₂
        self.v = nn.Linear(hidden_dim, 1)            # Vᵗ

    def forward(self, query, keys, values):
        """
        query:  (B, query_dim)               -> decoder'ın o anki durumu
        keys:   (B, seq_len, key_dim)        -> encoder'dan gelen gizli durumlar
        values: (B, seq_len, value_dim)      -> genelde keys ile aynı
        """
        B, T, _ = keys.size()

        query_proj = self.W_q(query).unsqueeze(1)   
        keys_proj = self.W_k(keys)                     

        energy = torch.tanh(query_proj + keys_proj)   
        scores = self.v(energy).squeeze(-1)            

        attn_weights = torch.softmax(scores, dim=1)    
        context = torch.bmm(attn_weights.unsqueeze(1), values).squeeze(1)  # (B, V)

        return context, attn_weights
    
class TemporalAttention(nn.Module):
    #Zamansal verilerde (video, sensor datası vs.) hangi zaman adımının daha önemli olduğunu öğrenir.
    def __init__(self, feature_dim):
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = torch.sqrt(torch.FloatTensor([feature_dim]))

    def forward(self, video_features):
       
        Q = self.query(video_features)  # [batch_size, num_frames, feature_dim]
        K = self.key(video_features)    # [batch_size, num_frames, feature_dim]
        V = self.value(video_features)  # [batch_size, num_frames, feature_dim]

        # Compute attention scores
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  
        attention_weights = F.softmax(attention_scores, dim=-1)  

        # Apply the attention weights to the value matrix
        attended_features = torch.bmm(attention_weights, V)  

        return attended_features, attention_weights
    
class SelfAttention(nn.Module):
    #Bir dizinin her elemanının diğerleriyle ilişkisini öğrenmesini sağlar. Her pozisyon, diğer tüm pozisyonlara dikkat verir.
    def __init__(self, n_embd,n_head,is_casual=False):
        super().__init__()
        assert n_embd % n_head == 0
        self.is_casual=is_casual

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd,n_embd)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout=nn.Dropout(0.2)
        

    def forward(self, x):
        B, T, C = x.size() 
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_casual) 
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)
        y = self.dropout(y)
        
        return y

#nn.MultiheadAttention(768)
#Farklı dikkat odaklarını paralel olarak öğrenmek için kullanılır. Özellikle self-attention ile birlikte kullanılır.



class DSC(nn.Sequential):
    def __init__(self,input,output):
        super().__init__(
            nn.Conv2d(input,input,kernel_size=3,padding=1,groups=input),
            nn.Conv2d(input,output,kernel_size=1)
        )
    def forward(self,x):
        for module in self:
            x=module(x)
        return x
    

if __name__=="__main__":

    N, C, H, W = 1, 256, 56, 56
    def count_parameters(model: nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    model = SelfAttention(56*56,4)

    print(count_parameters(model))
