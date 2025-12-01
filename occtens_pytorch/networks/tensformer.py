import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x
    
class Attention(nn.Module):
    def __init__(
        self, 
        dim,
        dim_head=64,
        num_heads=8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.num_heads = num_heads
        
        inner_dim = dim_head * num_heads
        self.q = nn.Linear(dim, inner_dim, bias=False)
        self.kv = nn.Linear(dim, inner_dim*2, bias=False)
        self.out = nn.Linear(inner_dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, context=None, attn_mask=None):
        x = self.norm(x)
        x_kv = context if context is not None else x

        q = self.q(x)
        k, v = self.kv(x_kv).chunk(2, dim=-1)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), (q, k, v))

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        if attn_mask is not None:
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)
        
        attn = sim.softmax(dim = -1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, mult):
        super().__init__()
        self.dim = dim
        self.mult = mult
        self.inner_dim = int(dim * mult * 2 / 3)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.inner_dim*2, bias = False),
            GEGLU(),
            nn.Linear(self.inner_dim, dim, bias = False)
        )
    def forward(self, x):
        return self.ff(x)

class Decoder(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        num_heads=8,
        ff_mult=4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)

        for _ in range():
            self.layers.append(nn.ModuleList([
            Attention(dim=dim, dim_head=dim_head, num_heads=num_heads),
            Attention(dim=dim, dim_head=dim_head, num_heads=num_heads),
            FeedForward(dim=dim, mult=ff_mult)
        ]))
        
    def forward(self, tokens, attn_mask_temporal, attn_mask_spatial):
        B, T, N, D = tokens.size()
        for temporal_attn, spatial_attn, ff in self.layers:
            x = rearrange(tokens, 'b t n d -> b (t n) d')
            x = x + temporal_attn(x, attn_mask=attn_mask_temporal)
            x = rearrange(x, 'b (t n) d -> b t n d', t=T)

            x_frame = rearrange(x, 'b t n d -> (b t) n d')
            x_frame = x_frame + spatial_attn(x_frame, attn_mask=attn_mask_spatial)
            x = rearrange(x_frame, '(b t) n d -> b t n d', t=T)

            x = x + ff(x)
            tokens = x

        return self.norm(tokens)

class TENSFormer(nn.Module):
    def __init__(
        self,
        dim,
        scene_tokenizer,
        motion_tokenizer,
        dim_head=64,
        num_heads=8,
        ff_mult=4,
    ):
        super().__init__()
        self.decoder = Decoder(dim, dim_head, num_heads, ff_mult)
        self.scene_tokenizer = scene_tokenizer
        self.motion_tokenizer = motion_tokenizer

    def forward(self, scene, motion, context=None):
        scene_token = self.scene_tokenizer(scene)
        motion_token = self.motion_tokenizer(motion)
        
        attn_mask_temporal = None
        attn_mask_spatial = None
        tokens = torch.cat([motion_token, scene_token], dim=-1)
        out = self.decoder(tokens, context=context, attn_mask_temporal=attn_mask_temporal, attn_mask_spatial=attn_mask_spatial)

        return out
            
class AutoRegressiveWrapper(nn.Module):
    def __init__(
        self,
        model
    ):
        super().__init__()
        self.model = model

    def forward(self, batch):
        pass