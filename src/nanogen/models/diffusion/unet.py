import torch
import math
import typing as ty

from einops import rearrange
from itertools import pairwise
from torch import nn
from smalldiffusion.model import (
    Attention, ModelMixin, CondSequential, SigmaEmbedderSinCos,
    CondEmbedderLabel
)
from nanoconfig import config
from . import CondEmbedder, DiffuserConfig, ModelConfig, Diffuser, SampleValue, CondValue

def Normalize(ch):
    return torch.nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6, affine=True)

def Upsample(ch, Conv):
    return nn.Sequential(
        nn.Upsample(scale_factor=2.0, mode='linear', align_corners=True),
        Conv(ch, ch, kernel_size=3, stride=1, padding=1),
    )

def Downsample(ch, Conv):
    return nn.Sequential(
        nn.ConstantPad1d((0, 1), 0),
        Conv(ch, ch, kernel_size=3, stride=2, padding=0),
    )


class ResnetBlock(nn.Module):
    def __init__(self, *, in_ch, out_ch=None, conv_shortcut=False,
                 dropout, temb_channels=512, Conv):
        super().__init__()
        self.in_ch = in_ch
        out_ch = in_ch if out_ch is None else out_ch
        self.out_ch = out_ch
        self.use_conv_shortcut = conv_shortcut

        self.layer1 = nn.Sequential(
            Normalize(in_ch),
            nn.SiLU(),
            Conv(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            torch.nn.Linear(temb_channels, out_ch),
        )
        self.layer2 = nn.Sequential(
            Normalize(out_ch),
            nn.SiLU(),
            torch.nn.Dropout(dropout),
            Conv(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )
        if self.in_ch != self.out_ch:
            kernel_stride_padding = (3, 1, 1) if self.use_conv_shortcut else (1, 1, 0)
            self.shortcut = Conv(in_ch, out_ch, *kernel_stride_padding)

    def forward(self, x, temb):
        h = x
        h = self.layer1(h)
        h = h + self.temb_proj(temb)[:, :, None]
        h = self.layer2(h)
        if self.in_ch != self.out_ch:
            x = self.shortcut(x)
        return x + h

class AttnBlock(nn.Module):
    def __init__(self, ch, num_heads=1):
        super().__init__()
        # Normalize input along the channel dimension
        self.norm = Normalize(ch)
        # Attention over D: (B, N, D) -> (B, N, D)
        self.attn = Attention(head_dim=ch // num_heads, num_heads=num_heads)
        # Apply 1x1 convolution for projection
        self.proj_out = nn.Conv1d(ch, ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        # temb is currently not used, but included for CondSequential to work
        B, C, L = x.shape
        h_ = self.norm(x)
        h_ = rearrange(h_, 'b c l -> b l c')
        h_ = self.attn(h_)
        h_ = rearrange(h_, 'b l c -> b c l')
        return x + self.proj_out(h_)

class Unet(Diffuser):
    def __init__(self, spatial_dims, in_ch, out_ch,
                 ch               = 128,
                 ch_mult          = (1,2,2,2),
                 embed_ch_mult    = 4,
                 num_res_blocks   = 2,
                 attn_resolutions = (16,),
                 dropout          = 0.1,
                 resample_with_conv = True,
                 sig_embed        = None,
                 cond_embed       = None,
                 ):
        super().__init__()
        self.ch = ch
        self.in_ch = in_ch
        self.spatial_dims = spatial_dims
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.temb_ch = self.ch * embed_ch_mult

        if spatial_dims == 1:
            Conv = nn.Conv1d
        elif spatial_dims == 2:
            Conv = nn.Conv2d
        elif spatial_dims == 3:
            Conv = nn.Conv3d
        else:
            raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")

        # Embeddings
        self.sig_embed = sig_embed or SigmaEmbedderSinCos(self.temb_ch)
        make_block = lambda in_ch, out_ch: ResnetBlock(
            in_ch=in_ch, out_ch=out_ch, temb_channels=self.temb_ch, dropout=dropout, Conv=Conv
        )
        self.cond_embed = cond_embed

        # Downsampling
        in_ch_dim = [ch * m for m in (1,)+tuple(ch_mult)]
        self.conv_in = Conv(in_ch, self.ch, kernel_size=3, stride=1, padding=1)
        self.downs = nn.ModuleList()
        block_in, block_out = 0, 0
        red_factor = 1
        for i, (block_in, block_out) in enumerate(pairwise(in_ch_dim)):
            down = nn.Module()
            down.blocks = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                block : list = [make_block(block_in, block_out)]
                if red_factor in attn_resolutions:
                    block.append(AttnBlock(block_out))
                down.blocks.append(CondSequential(*block))
                block_in = block_out
            if i < self.num_resolutions - 1: # Not last iter
                down.downsample = Downsample(block_in, Conv=Conv)
                red_factor *= 2
            self.downs.append(down)

        # Middle
        self.mid = CondSequential(
            make_block(block_in, block_in),
            AttnBlock(block_in),
            make_block(block_in, block_in)
        )

        # Upsampling
        self.ups = nn.ModuleList()
        for i_level, (block_out, next_skip_in) in enumerate(pairwise(reversed(in_ch_dim))):
            up = nn.Module()
            up.blocks = nn.ModuleList()
            skip_in = block_out
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = next_skip_in
                block = [make_block(block_in+skip_in, block_out)]
                if red_factor in attn_resolutions:
                    block.append(AttnBlock(block_out)) # type: ignore
                up.blocks.append(CondSequential(*block))
                block_in = block_out
            if i_level < self.num_resolutions - 1: # Not last iter
                up.upsample = Upsample(block_in, Conv=Conv) # type: ignore
                red_factor = red_factor // 2
            self.ups.append(up)

        # Out
        self.out_layer = nn.Sequential(
            Normalize(block_in), # type: ignore
            nn.SiLU(),
            Conv(block_in, out_ch, # type: ignore
                kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, sigma, cond=None):
        # Turn B T C -> B C T
        x = torch.transpose(x, -1, -2)
        # Embeddings
        emb = self.sig_embed(x.shape[0], sigma.squeeze())
        if cond is not None:
            assert self.cond_embed is not None
            emb += self.cond_embed(cond)
        # downsampling
        hs = [self.conv_in(x)]
        for down in self.downs:
            for block in down.blocks: # type: ignore
                h = block(hs[-1], emb)
                hs.append(h)
            if hasattr(down, 'downsample'):
                hs.append(down.downsample(hs[-1])) # type: ignore

        # middle
        h = self.mid(hs[-1], emb)

        # upsampling
        for up in self.ups:
            for block in up.blocks: # type: ignore
                h = block(torch.cat([h, hs.pop()], dim=1), emb)
            if hasattr(up, 'upsample'):
                h = up.upsample(h) # type: ignore

        # out
        out = self.out_layer(h)
        # Turn B C T -> B T C
        out = torch.transpose(out, -1, -2)
        return out

@config(variant="unet")
class UnetConfig(DiffuserConfig):
    base_channels: int = 64
    channel_mults: ty.Sequence[int] = (1, 2, 2, 2)
    embed_channel_mult: int = 4
    num_res_blocks: int = 2
    attn_resolutions: ty.Sequence[int] = (8,)
    dropout: float = 0.1
    resample_with_conv: bool = True

    def create(self, sample_structure: SampleValue, cond_structure: CondValue) -> Unet:
        assert isinstance(sample_structure, torch.Tensor), "Can only use Unet for single tensor samples."
        embed_features = self.base_channels * self.embed_channel_mult
        return Unet(
            spatial_dims=sample_structure.ndim - 1,
            in_ch=sample_structure.shape[-1],
            out_ch=sample_structure.shape[-1],
            ch=self.base_channels,
            ch_mult=self.channel_mults,
            embed_ch_mult=self.embed_channel_mult,
            num_res_blocks=self.num_res_blocks,
            attn_resolutions=self.attn_resolutions,
            dropout=self.dropout,
            resample_with_conv=self.resample_with_conv,
            sig_embed=SigmaEmbedderSinCos(embed_features),
            cond_embed=CondEmbedder(cond_structure, embed_features)
        )
