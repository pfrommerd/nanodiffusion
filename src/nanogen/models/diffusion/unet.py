import torch
import math
import functools
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

def Upsample(ch):
    return nn.Sequential(
        nn.Upsample(scale_factor=2.0, mode='nearest'),
        nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
    )

def Downsample(ch):
    return nn.Sequential(
        nn.ConstantPad2d((0, 1, 0, 1), 0),
        nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=0)
    )

class ResnetBlock(nn.Module):
    def __init__(self, *, in_ch, out_ch=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_ch = in_ch
        out_ch = in_ch if out_ch is None else out_ch
        self.out_ch = out_ch
        self.use_conv_shortcut = conv_shortcut

        self.layer1 = nn.Sequential(
            Normalize(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            torch.nn.Linear(temb_channels, out_ch),
        )
        self.layer2 = nn.Sequential(
            Normalize(out_ch),
            nn.SiLU(),
            torch.nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )
        if self.in_ch != self.out_ch:
            kernel_stride_padding = (3, 1, 1) if self.use_conv_shortcut else (1, 1, 0)
            self.shortcut = nn.Conv2d(in_ch, out_ch, *kernel_stride_padding)

    def forward(self, x, temb):
        h = x
        h = self.layer1(h)
        embed = self.temb_proj(temb)
        while embed.ndim < h.ndim:
            embed = embed.unsqueeze(-1)
        h = h + embed
        h = self.layer2(h)
        if self.in_ch != self.out_ch:
            x = self.shortcut(x)
        return x + h

class AttnBlock(nn.Module):
    def __init__(self, ch, *, num_heads=1):
        super().__init__()
        # Normalize input along the channel dimension
        self.norm = Normalize(ch)
        # Attention over D: (B, N, D) -> (B, N, D)
        self.attn = Attention(head_dim=ch // num_heads, num_heads=num_heads)
        # Apply 1x1 convolution for projection
        self.proj_out = nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        # temb is currently not used, but included for CondSequential to work
        h_ = self.norm(x)
        spatial_dims = h_.shape[2:]
        h_ = h_.view(h_.shape[0], h_.shape[1], math.prod(spatial_dims))
        h_ = h_.transpose(1, -1) # move channel to last dimension
        h_ = self.attn(h_)
        h_ = h_.transpose(1, -1) # move channel back to first dimension
        h_ = h_.view(h_.shape[0], h_.shape[1], *spatial_dims)
        return x + self.proj_out(h_)

def alpha(sigma):
    return (1/(1+sigma**2)).sqrt()

class Unet(ModelMixin, Diffuser):
    def __init__(self, in_dim, in_ch, out_ch,
                 ch               = 128,
                 ch_mult          = (1,2,2,2),
                 embed_ch_mult    = 4,
                 num_res_blocks   = 2,
                 attn_resolutions = (2,),
                 dropout          = 0.1,
                 sig_embed        = None,
                 cond_embed       = None):
        super().__init__()

        self.ch = ch
        self.in_dim = in_dim
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.input_dims = (in_dim, in_dim, ch)
        self.temb_ch = self.ch * embed_ch_mult

        # Embeddings
        self.sig_embed = sig_embed or SigmaEmbedderSinCos(self.temb_ch)
        make_block = lambda in_ch, out_ch: ResnetBlock(
            in_ch=in_ch, out_ch=out_ch, temb_channels=self.temb_ch, dropout=dropout
        )
        self.cond_embed = cond_embed

        # Downsampling
        in_ch_dim = [ch * m for m in (1,)+tuple(ch_mult)]
        self.conv_in = nn.Conv2d(in_ch, self.ch, kernel_size=3, stride=1, padding=1)
        self.downs = nn.ModuleList()
        block_in, block_out = 0, 0
        for i, (block_in, block_out) in enumerate(pairwise(in_ch_dim)):
            down = nn.Module()
            down.blocks = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                block = [make_block(block_in,block_out)]
                if i in attn_resolutions:
                    block.append(AttnBlock(block_out))
                down.blocks.append(CondSequential(*block))
                block_in = block_out
            if i < self.num_resolutions - 1: # Not last iter
                down.downsample = Downsample(block_in)
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
                block: list = [make_block(block_in+skip_in, block_out)]
                if (self.num_resolutions - 1 - i_level) in attn_resolutions:
                    block.append(AttnBlock(block_out))
                up.blocks.append(CondSequential(*block))
                block_in = block_out
            if i_level < self.num_resolutions - 1: # Not last iter
                up.upsample = Upsample(block_in)
            self.ups.append(up)

        # Out
        self.out_layer = nn.Sequential(
            Normalize(block_in),
            nn.SiLU(),
            nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
        )

    def fwd_emb(self, x, emb, cond):
        assert x.shape[2] == x.shape[3] == self.in_dim

        if self.cond_embed is not None:
            assert cond is not None and x.shape[0] == cond.shape[0], \
                'Conditioning must have same batches as x!'
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
        return self.out_layer(h)

    def forward(self, x, sigma, cond=None):
        a = alpha(sigma).reshape(sigma.shape + (1,) * (len(x.shape) - 1))
        emb = self.sig_embed(x.shape[0], sigma.squeeze())
        # transpose from NHWC to NCHW
        x = rearrange(x, 'b h w c -> b c h w')
        x = a*x
        x = self.fwd_emb(x, emb, cond)
        x = rearrange(x, 'b c h w -> b h w c')
        return x

@config(variant="unet")
class UnetConfig(DiffuserConfig):
    base_channels: int = 64
    channel_mults: ty.Sequence[int] = (1, 2, 2, 2)
    embed_channel_mult: int = 4
    num_res_blocks: int = 2
    attn_resolutions: ty.Sequence[int] = (2,)
    dropout: float = 0.1

    def create(self, sample_structure: SampleValue, cond_structure: CondValue) -> Unet:
        assert isinstance(sample_structure, torch.Tensor), "Can only use Unet for single tensor samples."
        embed_features = self.base_channels * self.embed_channel_mult
        in_dim = sample_structure.shape[-2]
        return Unet(
            in_dim=in_dim,
            in_ch=sample_structure.shape[-1],
            out_ch=sample_structure.shape[-1],
            ch=self.base_channels,
            ch_mult=self.channel_mults,
            embed_ch_mult=self.embed_channel_mult,
            num_res_blocks=self.num_res_blocks,
            attn_resolutions=self.attn_resolutions,
            dropout=self.dropout,
            sig_embed=SigmaEmbedderSinCos(embed_features),
            cond_embed=CondEmbedder(cond_structure, embed_features)
        )
