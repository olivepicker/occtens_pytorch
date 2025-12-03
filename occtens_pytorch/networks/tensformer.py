import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

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
        
        B = x.size(0)
        if (attn_mask is not None) & (len(attn_mask.size())==2):
            attn_mask = repeat(attn_mask, 'h w -> b 1 h w', b=B)

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
        num_layers=4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)

        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
            Attention(dim=dim, dim_head=dim_head, num_heads=num_heads),
            Attention(dim=dim, dim_head=dim_head, num_heads=num_heads),
            FeedForward(dim=dim, mult=ff_mult)
        ]))
        
    def forward(self, tokens, num_frames, attn_mask_temporal, attn_mask_spatial, context=None):
        bos, x = tokens[:, :1, :], tokens[:, 1:, :]
 
        for temporal_attn, spatial_attn, ff in self.layers:
            x = x + temporal_attn(x, attn_mask=attn_mask_temporal)
            x = rearrange(x, 'b (f t) d -> b f t d', f=num_frames)

            x_frame = rearrange(x, 'b f t d -> (b f) t d', f=num_frames)
            x_frame = x_frame + spatial_attn(x_frame, attn_mask=attn_mask_spatial)
            x = rearrange(x_frame, '(b f) t d -> b (f t) d', f=num_frames)

            x = x + ff(x)

        tokens = torch.cat([bos, x], dim=1) 
        return self.norm(tokens)

class TENSFormer(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        num_heads=8,
        num_layers=4,
        ff_mult=4,
    ):
        super().__init__()
        self.bos_token = nn.Parameter(torch.randn(1, 1, dim))
        self.decoder = Decoder(dim, dim_head, num_heads, ff_mult, num_layers)

    def forward(
        self, 
        scene_tokens, 
        motion_tokens, 
        lengths, 
        context=None
    ):
        B, F = scene_tokens.shape[:2]
        device = scene_tokens.device
        ends = torch.cumsum(lengths, dim=0)
        max_cols_per_scale = ends - 1
        max_col_for_row = torch.repeat_interleave(max_cols_per_scale, lengths)

        N = int(max_col_for_row.shape[0])
        col_idx = torch.arange(N, device=device)
        scale_mask = col_idx.unsqueeze(0) <= max_col_for_row.unsqueeze(1)
        time_idx = torch.arange(F, device=device)
        time_mask = time_idx.unsqueeze(1) >= time_idx.unsqueeze(0)
        attn_mask_temporal = time_mask[:, :, None, None] & scale_mask[None, None, :, :]
        attn_mask_temporal = attn_mask_temporal.view(F * N, F * N)
        attn_mask_spatial = scale_mask

        tokens = torch.cat([motion_tokens, scene_tokens], dim=2)
        bos_token = self.bos_token.expand(B, 1, -1)
        tokens = torch.cat([bos_token, rearrange(tokens, 'b f t d -> b (f t) d')], dim=1)

        embedding = self.decoder(
            tokens,
            num_frames=F,
            attn_mask_temporal=attn_mask_temporal, 
            attn_mask_spatial=attn_mask_spatial,
            context=context
        )

        return embedding