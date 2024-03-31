import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint
from timm.models.layers import DropPath, to_2tuple
from mmcv.cnn.bricks import ConvModule, build_activation_layer, build_norm_layer
from collections import OrderedDict
from einops import rearrange
import random as rd

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768,
                 norm_layer=dict(type='BN2d'), act_cfg=None ):
        super().__init__()
        self.proj = ConvModule(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding,
            norm_cfg=norm_layer, act_cfg=act_cfg,
        )

    def forward(self, x):
        return self.proj(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=1, qk_scale=None, attn_drop=0, sr_ratio=1, ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                ConvModule(dim, dim, kernel_size=sr_ratio + 3, stride=sr_ratio, padding=(sr_ratio + 3) // 2,
                           groups=dim, bias=False, norm_cfg=dict(type='BN2d'), act_cfg=dict(type='GELU')),
                ConvModule(dim, dim, kernel_size=1, groups=dim, bias=False, norm_cfg=dict(type='BN2d'), act_cfg=None, )
            )
        else:
            self.sr = nn.Identity()
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, relative_pos_enc=None):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        kv = self.sr(x)
        kv = self.local_conv(kv) + kv
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)
        k = k.reshape(B, self.num_heads, C // self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        attn = (q @ k) * self.scale
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = F.interpolate(relative_pos_enc, size=attn.shape[2:], mode='bicubic', align_corners=False)
            attn = attn + relative_pos_enc
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2)
        return x.reshape(B, C, H, W)


class DynamicConv2d(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=4,
                 num_groups=2,
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.proj = nn.Sequential(
            ConvModule(dim, dim // reduction_ratio, kernel_size=1, norm_cfg=dict(type='BN2d'),
                       act_cfg=dict(type='GELU'), ),
            nn.Conv2d(dim // reduction_ratio, dim * num_groups, kernel_size=1),
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        scale = self.proj(self.pool(x)).reshape(B, self.num_groups, C, self.K, self.K)
        scale = torch.softmax(scale, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, self.K)

        if self.bias is not None:
            scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            scale = torch.softmax(scale.reshape(B, self.num_groups, C), dim=1)
            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None

        x = F.conv2d(x.reshape(1, -1, H, W), weight=weight, padding=self.K // 2, groups=B * C, bias=bias)

        return x.reshape(B, C, H, W)


class HybridTokenMixer(nn.Module):
    def __init__(self, dim, kernel_size=3, num_groups=2, num_heads=1, sr_ratio=1, reduction_ratio=8):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."

        self.local_unit = DynamicConv2d(dim=dim // 2, kernel_size=kernel_size, num_groups=num_groups)
        self.global_unit = Attention(dim=dim // 2, num_heads=num_heads, sr_ratio=sr_ratio)

        inner_dim = max(16, dim // reduction_ratio)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x, relative_pos_enc=None):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = self.local_unit(x1)
        x2 = self.global_unit(x2, relative_pos_enc)
        x = torch.cat([x1, x2], dim=1)
        x = self.proj(x) + x
        return x


class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels, kernel_size=scale[i], padding=scale[i] // 2, groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0, ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            build_activation_layer(act_cfg),
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = MultiScaleDWConv(hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x) + x
        x = self.norm(self.act(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1) * init_value, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x


class Block(nn.Module):
    def __init__(self, dim=64, kernel_size=3, sr_ratio=1, num_groups=2, num_heads=1, mlp_ratio=4,
                 norm_cfg=dict(type='GN', num_groups=1), act_cfg=dict(type='GELU'), drop=0, drop_path=0,
                 layer_scale_init_value=1e-5, grad_checkpoint=False):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.token_mixer = HybridTokenMixer(dim, kernel_size=kernel_size, num_groups=num_groups,
                                            num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_cfg=act_cfg, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale(dim, layer_scale_init_value)
            self.layer_scale_2 = LayerScale(dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()

    def _forward_impl(self, x, relative_pos_enc=None):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.layer_scale_1(self.token_mixer(self.norm1(x), relative_pos_enc)))
        x = x + self.drop_path(self.layer_scale_2(self.mlp(self.norm2(x))))
        return x

    def forward(self, x, relative_pos_enc=None):
        if self.grad_checkpoint and x.requires_grad:
            x = checkpoint.checkpoint(self._forward_impl, x, relative_pos_enc)
        else:
            x = self._forward_impl(x, relative_pos_enc)
        return x


def basic_blocks(dim, index, layers, kernel_size=3, num_groups=2, num_heads=1, sr_ratio=1, mlp_ratio=4,
                 norm_cfg=dict(type='GN', num_groups=1), act_cfg=dict(type='GELU'), drop_rate=0,
                 drop_path_rate=0, layer_scale_init_value=1e-5, grad_checkpoint=False):
    blocks = nn.ModuleList()
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
                block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(
            Block(dim, kernel_size=kernel_size, num_groups=num_groups, num_heads=num_heads, sr_ratio=sr_ratio,
                mlp_ratio=mlp_ratio, norm_cfg=norm_cfg, act_cfg=act_cfg, drop=drop_rate, drop_path=block_dpr,
                layer_scale_init_value=layer_scale_init_value, grad_checkpoint=grad_checkpoint)
            )
    return blocks

class PatchExpand(nn.Module):
    def __init__(self, in_chan, out_dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = in_chan
        self.expand = nn.Linear(in_chan, 4 * out_dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(2 * out_dim // dim_scale)

    def forward(self, x):
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        _, H, W, _ = x.shape
        x = self.norm(x)

        x = rearrange(x, 'b h w c-> b c h w')

        return x


class HCTRNet_encoder(nn.Module):
    def __init__(self, image_size=256, norm_cfg=dict(type='GN', num_groups=1), act_cfg=dict(type='GELU'),
                 in_chans=3, drop_rate=0, drop_path_rate=0, grad_checkpoint=False, checkpoint_stage=[0] * 4,
                 num_classes=1, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.grad_checkpoint = grad_checkpoint

        depths = [2, 2, 6, 2]
        embed_dims = [64, 128, 320, 512]
        kernel_size = [7, 7, 7, 7]
        num_groups = [2, 2, 2, 2]
        sr_ratio = [8, 4, 2, 1]
        num_heads = [1, 2, 5, 8]
        mlp_ratios = [8, 8, 4, 4]  # [4, 4, 4, 4]
        layer_scale_init_value = 1e-5

        if not grad_checkpoint:
            checkpoint_stage = [0] * 4

        self.patch_embed1 = PatchEmbed(patch_size=7, stride=4, padding=3, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(patch_size=3, stride=2, padding=1, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(patch_size=3, stride=2, padding=1, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(patch_size=3, stride=2, padding=1, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.relative_pos_enc = []
        self.pos_enc_record = []
        image_size = to_2tuple(image_size)
        image_size = [math.ceil(image_size[0] / 4), math.ceil(image_size[1] / 4)]
        for i in range(4):
            num_patches = image_size[0] * image_size[1]
            sr_patches = math.ceil(image_size[0] / sr_ratio[i]) * math.ceil(image_size[1] / sr_ratio[i])
            self.relative_pos_enc.append(nn.Parameter(torch.zeros(1, num_heads[i], num_patches, sr_patches), requires_grad=True))
            self.pos_enc_record.append([image_size[0], image_size[1], math.ceil(image_size[0] / sr_ratio[i]),
                                        math.ceil(image_size[1] / sr_ratio[i]), ])
            image_size = [math.ceil(image_size[0] / 2), math.ceil(image_size[1] / 2)]
        self.relative_pos_enc = nn.ParameterList(self.relative_pos_enc)

        # set the main block in network
        former_layers = []
        for i in range(len(depths)):
            former_layer = basic_blocks(embed_dims[i], i, depths, kernel_size=kernel_size[i], num_groups=num_groups[i],
                num_heads=num_heads[i], sr_ratio=sr_ratio[i], mlp_ratio=mlp_ratios[i], norm_cfg=norm_cfg,
                act_cfg=act_cfg, drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value, grad_checkpoint=checkpoint_stage[i])
            former_layers.append(former_layer)
            if i >= len(depths) - 1:
                break
        self.former_layers = nn.ModuleList(former_layers)

        # add a norm layer for each output
        self.norm_layer1 = build_norm_layer(norm_cfg, embed_dims[0])[1]
        self.norm_layer2 = build_norm_layer(norm_cfg, embed_dims[1])[1]
        self.norm_layer3 = build_norm_layer(norm_cfg, embed_dims[2])[1]
        self.norm_layer4 = build_norm_layer(norm_cfg, embed_dims[3])[1]

    def forward(self, x):
        # layer1
        x1 = self.patch_embed1(x)
        for blk in self.former_layers[0]:
            x1 = blk(x1, self.relative_pos_enc[0])
        out1 = self.norm_layer1(x1)

        # layer2
        x2 = self.patch_embed2(out1)
        for blk in self.former_layers[1]:
            x2 = blk(x2, self.relative_pos_enc[1])
        out2 = self.norm_layer2(x2)

        # layer3
        x3 = self.patch_embed3(out2)
        for blk in self.former_layers[2]:
            x3 = blk(x3, self.relative_pos_enc[2])
        out3 = self.norm_layer3(x3)

        # layer4
        x4 = self.patch_embed4(out3)
        for blk in self.former_layers[3]:
            x4 = blk(x4, self.relative_pos_enc[3])
        out4 = self.norm_layer4(x4)

        return out1, out2, out3, out4


class HCTRNet_decoder(nn.Module):
    def __init__(self, block_id, dim, image_size=256, norm_cfg=dict(type='GN', num_groups=1), act_cfg=dict(type='GELU'),
                 in_chans=3, drop_rate=0, drop_path_rate=0, grad_checkpoint=False, checkpoint_stage=[0] * 4,
                 num_classes=1, **kwargs):
        super(HCTRNet_decoder, self).__init__()
        self.num_classes = num_classes
        self.grad_checkpoint = grad_checkpoint

        depths = [2, 2, 6, 2]
        kernel_size = [7, 7, 7, 7]
        num_groups = [2, 2, 2, 2]
        sr_ratio = [8, 4, 2, 1]
        num_heads = [1, 2, 4, 8]
        mlp_ratios = [8, 8, 4, 4]  # [4, 4, 4, 4]
        layer_scale_init_value = 1e-5
        self.block_id = block_id

        i_range = self.block_id - 1

        if not grad_checkpoint:
            checkpoint_stage = [0] * 4

        image_size = to_2tuple(image_size)
        image_size = [math.ceil(image_size[0] / 4), math.ceil(image_size[1] / 4)]
        num_patches = image_size[0] * image_size[1]
        sr_patches = math.ceil(image_size[0] / sr_ratio[i_range]) * math.ceil(image_size[1] / sr_ratio[i_range])

        self.relative_pos_enc = nn.Parameter(torch.zeros(1, num_heads[i_range], num_patches, sr_patches), requires_grad=True)

        former_layers = []
        former_layer = basic_blocks(dim, i_range, depths, kernel_size=kernel_size[i_range], num_groups=num_groups[i_range],
                          num_heads=num_heads[i_range], sr_ratio=sr_ratio[i_range], mlp_ratio=mlp_ratios[i_range], norm_cfg=norm_cfg,
                          act_cfg=act_cfg, drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                          layer_scale_init_value=layer_scale_init_value, grad_checkpoint=checkpoint_stage[i_range])
        former_layers.append(former_layer)
        self.layer = nn.ModuleList(former_layers)

        # add a norm layer for each output
        self.norm_layer = build_norm_layer(norm_cfg, dim)[1]

    def forward(self, x):
        # layer
        for blk in self.layer[0]:
            x = blk(x, self.relative_pos_enc)
        out = self.norm_layer(x)

        return out

# Adaptive Feature Fusion Decoder
class AFFH(nn.Module):
    def __init__(self, inter_dim, embed_dims=[64, 128, 320, 512], rfb=False):
        super(AFFH, self).__init__()
        self.inter_dim = inter_dim

        self.conv1 = self.add_conv(embed_dims[0], self.inter_dim, 1, 1)
        self.conv2 = self.add_conv(embed_dims[1], self.inter_dim, 1, 1)
        self.conv3 = self.add_conv(embed_dims[2], self.inter_dim, 1, 1)
        self.conv4 = self.add_conv(embed_dims[3], self.inter_dim, 1, 1)

        compress_c = 16

        self.weight_1 = self.add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_2 = self.add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_3 = self.add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_4 = self.add_conv(self.inter_dim, compress_c, 1, 1)

        self.weights = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)

    def add_conv(self, in_ch, out_ch, ksize, stride, leaky=True):
        stage = nn.Sequential()
        pad = (ksize - 1) // 2
        stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                           out_channels=out_ch, kernel_size=ksize, stride=stride,
                                           padding=pad, bias=False))
        stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
        if leaky:
            stage.add_module('leaky', nn.LeakyReLU(0.1))
        else:
            stage.add_module('relu6', nn.ReLU6(inplace=True))
        return stage

    def forward(self, x1, x2, x3, x4):
        x1_resized = self.conv1(x1)

        x2 = self.conv2(x2)
        x2_resized = F.interpolate(x2, scale_factor=2, mode='bilinear')

        x3 = self.conv3(x3)
        x3_resized = F.interpolate(x3, scale_factor=4, mode='bilinear')

        x4 = self.conv4(x4)
        x4_resized = F.interpolate(x4, scale_factor=8, mode='bilinear')

        x1_weight = self.weight_1(x1_resized)
        x2_weight = self.weight_2(x2_resized)
        x3_weight = self.weight_3(x3_resized)
        x4_weight = self.weight_4(x4_resized)
        out_weights = torch.cat((x1_weight, x2_weight, x3_weight, x4_weight), 1)
        out_weights = self.weights(out_weights)
        out_weights = F.softmax(out_weights, dim=1)

        out = x1_resized * out_weights[:, 0:1, :, :] + \
              x2_resized * out_weights[:, 1:2, :, :] + \
              x3_resized * out_weights[:, 2:3, :, :] + \
              x4_resized * out_weights[:, 3:, :, :]

        return out

class HCTRNet(nn.Module):
    def __init__(self, image_size=(256, 256), num_classes=1, embed_dims=[512, 320, 128, 64], decoder_dim=128, drop_rate=0.):
        super(HCTRNet, self).__init__()
        self.image_size = image_size
        self.backbone = HCTRNet_encoder(image_size)  # [64, 128, 320, 512]

        self.convblock = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
            )

        self.decoder1 = HCTRNet_decoder(4, embed_dims[0], image_size[0] // 8)
        self.decoder2 = HCTRNet_decoder(3, embed_dims[1], image_size[0] // 4)
        self.decoder3 = HCTRNet_decoder(2, embed_dims[2], image_size[0] // 2)
        self.decoder4 = HCTRNet_decoder(1, embed_dims[3], image_size[0])

        self.patch_expand1 = PatchExpand(1024, embed_dims[0])
        self.patch_expand2 = PatchExpand(embed_dims[0], embed_dims[1])
        self.patch_expand3 = PatchExpand(embed_dims[1], embed_dims[2])
        self.patch_expand4 = PatchExpand(embed_dims[2], embed_dims[3])

        # if add head then embed_dims[3]->decoder_dim
        self.affh = AFFH(decoder_dim)
        # if add head then embed_dims[3]->decoder_dim
        self.linear_pred = nn.Sequential(
            nn.ConvTranspose2d(decoder_dim, 32, 4, 4),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, num_classes, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)

        center = self.convblock(x4)  # [-1, 1024, H/64, W/64]

        c1 = self.decoder1(self.patch_expand1(center) + x4)
        c2 = self.decoder2(self.patch_expand2(c1) + x3)
        c3 = self.decoder3(self.patch_expand3(c2) + x2)
        c4 = self.decoder4(self.patch_expand4(c3) + x1)

        out = self.affh(c4, c3, c2, c1)
        
        # if add head then c4->out
        x = self.linear_pred(out)
        return x


if __name__ == "__main__":
    x = torch.randn((4, 3, 256, 256))
    model = HCTRNet()
    y = model(x)
    print(y.shape)
    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(model, input_res=(3, 256, 256), as_strings=True,
                                              print_per_layer_stat=False)

    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)
