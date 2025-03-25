import torch
import torch.nn as nn
from extensions.chamfer_dist import ChamferDistanceL1
from .model_utils import MLP_CONV, Transformer, PointNet_SA_Module_KNN
from .build import MODELS

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):

        B, N, C = x.shape
        _, NK, _ = y.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  #  B, H, N, C
        k = self.k(y).reshape(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  #  B, H, NK, C
        v = self.v(y).reshape(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  #  B, H, NK, C

        attn = (q @ k.transpose(-2, -1)) * self.scale #  B, H, N, NK
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossFormer(nn.Module):

    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, drop_path=0.1):
        super().__init__()
        self.bn1 = nn.LayerNorm(dim)
        self.bn2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        self.bn3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x, y):
        short_cut = x
        x = self.bn1(x)
        y = self.bn2(y)
        x = self.attn(query=x, key=y, value=y)[0]
        x = short_cut + self.drop_path(x)
        x = x + self.drop_path(self.ffn(self.bn3(x)))
        return x

class LSTNet(nn.Module):
    def __init__(self, out_dim=512):
        super(LSTNet, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(128, dim=64)
        self.expanding = MLP_CONV(in_channel=128, layer_dims=[256, out_dim])     
        
        self.mlp = nn.Sequential(
                nn.Linear(512*2, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 9+3)
        )

    def forward(self, point_cloud):
        b = point_cloud.shape[0]
        l0_xyz = point_cloud
        l0_points = point_cloud

        # get the key points and its features
        keypoints, keyfeatures, _ = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)   
        keyfeatures = self.transformer_1(keyfeatures, keypoints) # B,128,512

        feat = self.expanding(keyfeatures)
        feat = feat.transpose(2, 1).contiguous()
        gf_feat = feat.max(dim=1, keepdim=True)[0]
        feat = torch.cat([feat, gf_feat.repeat(1, feat.size(2), 1)], dim=-1) # B,640,512

        ret = self.mlp(feat)   
        R = ret[:, :, :9].view(b, 512, 3, 3)
        T = ret[:, :, 9:]
        symmetry_points = torch.matmul(keypoints.transpose(2, 1).contiguous().unsqueeze(2), R).view(b, 512, 3)
        symmetry_points = symmetry_points + T
        symmetry_points = symmetry_points.transpose(2, 1).contiguous()
        coarse = torch.cat([symmetry_points, keypoints], dim=-1) # B, 1024, 3
        return coarse, symmetry_points, keyfeatures


class Fusion(nn.Module):
    def __init__(self, in_channel=512):
        super(Fusion, self).__init__()
        
        self.corssformer_1 = CrossFormer(in_channel, in_channel, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0)
        self.corssformer_2 = CrossFormer(in_channel, in_channel, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0)
    
    def forward(self, feat_x, feat_y):
        # cross attention
        feat = self.corssformer_1(feat_x, feat_y)
        
        # self attention
        feat = self.corssformer_2(feat, feat)
        return feat

class SGFormer(nn.Module):
    def __init__(self, gf_dim=512, up_factor=2):
        super(SGFormer, self).__init__()
        self.up_factor = up_factor
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_gf = MLP_CONV(in_channel=gf_dim, layer_dims=[256, 128])
        self.mlp_2 = MLP_CONV(in_channel=256, layer_dims=[256, 128])
        self.transformer = Transformer(in_channel=128, dim=64)
        
        self.expand_dim_1 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_2 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_3 = MLP_CONV(in_channel=128, layer_dims=[128, 256])

        self.fusion_1 = Fusion(in_channel=256)
        self.fusion_2 = Fusion(in_channel=256)
        
        self.mlp_fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512)
        )
        self.fusion_3 = Fusion(in_channel=512)

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 3 * self.up_factor)
        )

    def forward(self, coarse, symmetry_feat, partial_feat):
        b, _, n = coarse.shape
        feat = self.mlp_1(coarse)
        feat_max = feat.max(dim=-1, keepdim=True)[0]
        feat= torch.cat([feat, feat_max.repeat(1, 1, feat.shape[-1])], dim=1)
        feat = self.mlp_2(feat)
        feat = self.transformer(feat, coarse)

        feat = self.expand_dim_1(feat)
        partial_feat = self.expand_dim_2(partial_feat)
        symmetry_feat = self.expand_dim_3(symmetry_feat)

        feat = feat.transpose(2, 1).contiguous()
        partial_feat = partial_feat.transpose(2, 1).contiguous()
        symmetry_feat = symmetry_feat.transpose(2, 1).contiguous()

        # partial part awareness
        feat_p = self.fusion_1(feat, partial_feat)
        # symmetric part awareness
        feat_s = self.fusion_2(feat, symmetry_feat) 
        # fusion feature
        feat = torch.cat([feat_p, feat_s], dim=-1)
        feat = self.mlp_fusion(feat)

        # self attention for upsampling
        feat = self.fusion_3(feat, feat)
        offset = self.fc(feat).view(b, -1, 3) # B, N * up_ratio, 3
        pcd_up = coarse.transpose(2, 1).contiguous().unsqueeze(dim=2).repeat(1, 1, self.up_factor, 1).view(b, -1, 3) + offset
        return pcd_up

class local_encoder(nn.Module):
    def __init__(self,out_channel=128):
        super(local_encoder, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2, layer_dims=[256, out_channel])
        self.transformer = Transformer(out_channel, dim=64)

    def forward(self,input):
        feat = self.mlp_1(input)
        feat = torch.cat([feat,torch.max(feat, 2, keepdim=True)[0].repeat((1, 1, feat.size(2)))], 1)
        feat = self.mlp_2(feat)
        feat = self.transformer(feat,input)

        return feat


@MODELS.register_module()
class SymmCompletion(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.up_factors = [int(i) for i in config.up_factors.split(',')]
        self.lstnet = LSTNet(out_dim=512)
        self.local_encoder = local_encoder(out_channel=128)
        self.sgformer_1 = SGFormer(gf_dim=512, up_factor=self.up_factors[0])
        self.sgformer_2 = SGFormer(gf_dim=512, up_factor=self.up_factors[1])
        self.include_input = config.include_input
        self.loss_func = ChamferDistanceL1()
        
    def get_loss(self, rets, gt):
        loss_list = []
        loss_total = 0
        for pcd in rets:
            loss = self.loss_func(pcd, gt)
            loss_list.append(loss)
            loss_total += loss
        return loss_total, loss_list[0], loss_list[-1], loss_list[0], loss_list[-1]

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: (B, N, 3)
        """
        coarse, symmetry_points, keyfeatures = self.lstnet(point_cloud.transpose(2,1).contiguous())
        feat_symmetry = self.local_encoder(symmetry_points) # B,128,512
        feat_partial = keyfeatures # B,128,512
        fine1 = self.sgformer_1(coarse, feat_symmetry, feat_partial)
        fine2 = self.sgformer_2(fine1.transpose(2,1).contiguous(), feat_symmetry, feat_partial)

        if self.include_input: 
            fine2 = torch.cat([fine2, point_cloud],dim=1).contiguous()

        rets = [coarse.transpose(2,1).contiguous(), fine1, fine2]
        self.pred_dense_point = rets[-1]

        return rets

