import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_
import math
import einops
#inspried from : https://github.com/SebastianJanampa/LINEA/blob/master/models/linea/attention_mechanism.py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")
    return (n & (n-1) == 0) and n != 0

def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights, total_num_points):
    # for debug and test only, need to use cuda version instead
    N_, S_, M_, D_ = value.shape  # [1, 3136, 4, 64]
    N_, Lq_, M_, L_, P_, _ = sampling_locations.shape  # [1, 3136, 4, 1, 4, 2]
    
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_value_list = []

    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        
        sampling_grid_l_ = sampling_locations[:, :, :, lid_, :, :]  # (N_, Lq_, M_, P_, 2)
        sampling_grid_l_ = (2 * sampling_grid_l_ - 1).transpose(1, 2).flatten(0, 1)  # (N_*M_, Lq_, P_, 2)

        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear',
                                          padding_mode='zeros',
                                          align_corners=False)
        sampling_value_list.append(sampling_value_l_)

    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, total_num_points)
    output = (torch.cat(sampling_value_list, dim=-1) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=1, n_heads=4, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model must be divisible by n_heads, but got {d_model} and {n_heads}')

        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            import warnings
            warnings.warn("You'd better set d_model so that each head has a power of 2 dimension.")

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        if isinstance(n_points, list):
            assert len(n_points) == n_levels
            self.num_points_list = n_points
        else:
            self.num_points_list = [n_points] * n_levels

        self.total_num_points = sum(self.num_points_list)

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.view(-1))

        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape

        value = self.value_proj(input_flatten).view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1).to("cuda")  # (L, 2)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]


        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights, self.total_num_points)
        return self.output_proj(output)
class Use_Def_att(nn.Module):
    """Some Information about Use_Def_att"""
    def __init__(self,d_model=256, n_levels=1, n_heads=4, n_points=4):
        super(Use_Def_att, self).__init__()
        self.model=MSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
        self.n_levels = 1
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        N, C, H, W = x.shape
        Lq = H * W
        residue=x
        input_flatten = x.flatten(2).transpose(1, 2)
        input_spatial_shapes = torch.tensor([[H, W]], dtype=torch.long).to(x.device)
        query = input_flatten.clone()

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=x.device),
            torch.arange(W, dtype=torch.float32, device=x.device),
            indexing='ij'
        )
        ref_points = torch.stack([grid_x / (W - 1), grid_y / (H - 1)], dim=-1)  # (H, W, 2)
        ref_points = ref_points.view(1, Lq, 1, 2).repeat(N, 1, self.n_levels, 1)  # (N, Lq, n_levels, 2)

        out = self.model(query, ref_points, input_flatten, input_spatial_shapes)
        out = out.transpose(1, 2).view(N, C, H, W)
        out=residue+out
        return out




if __name__ == "__main__":
    N, C, H, W = 1, 256,56,56
    def count_parameters(model: nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    model = Use_Def_att(256).to("cuda")
    print("Toplam eğitimlenebilir parametre sayısı:", count_parameters(model))
    x = torch.rand(N, C, H, W).to("cuda")

    out=model(x)
    print(out.shape)
