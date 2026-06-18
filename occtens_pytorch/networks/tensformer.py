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
        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = repeat(attn_mask, 'h w -> b 1 h w', b=B)

        x = self.norm(x)
        x_kv = context if context is not None else x

        q = self.q(x)
        k, v = self.kv(x_kv).chunk(2, dim=-1)        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), (q, k, v))
        q = q * self.scale

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
        dim_head=32,
        num_heads=4,
        ff_mult=4,
        num_layers=4,
        spatial_mode="block",   # "full" or "block"
    ):
        super().__init__()

        assert spatial_mode in ["full", "block"]
        self.spatial_mode = spatial_mode

        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)

        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, dim_head=dim_head, num_heads=num_heads),
                Attention(dim=dim, dim_head=dim_head, num_heads=num_heads),
                FeedForward(dim=dim, mult=ff_mult)
            ]))

    def forward(
        self,
        tokens,
        num_frames,
        lengths,
        attn_mask_spatial=None,
        guidance=None,
    ):
        bos, x = tokens[:, :1, :], tokens[:, 1:, :]

        B = x.shape[0]
        device = x.device

        if guidance is not None:
            x = x + guidance[:, 1:, :]

        if not torch.is_tensor(lengths):
            lengths = torch.tensor(lengths, device=device, dtype=torch.long)
        else:
            lengths = lengths.to(device=device, dtype=torch.long)

        N = int(lengths.sum().item())

        if x.size(1) != num_frames * N:
            raise RuntimeError(
                f"x length mismatch: x.size(1)={x.size(1)}, "
                f"num_frames * N={num_frames * N}"
            )

        time_idx = torch.arange(num_frames, device=device)
        attn_mask_time = time_idx[:, None] >= time_idx[None, :]

        ends = torch.cumsum(lengths, dim=0)
        starts = torch.cat(
            [torch.zeros(1, device=device, dtype=torch.long), ends[:-1]],
            dim=0
        )

        for temporal_attn, spatial_attn, ff in self.layers:
            x_4d = rearrange(x, 'b (f t) d -> b f t d', f=num_frames, t=N)

            temporal_out = torch.zeros_like(x_4d)

            for start, end in zip(starts.tolist(), ends.tolist()):
                x_s = x_4d[:, :, start:end, :]  # (B, F, L, D)
                L = end - start

                x_s = rearrange(x_s, 'b f l d -> (b l) f d')

                y_s = temporal_attn(
                    x_s,
                    attn_mask=attn_mask_time
                )

                y_s = rearrange(
                    y_s,
                    '(b l) f d -> b f l d',
                    b=B,
                    l=L
                )

                temporal_out[:, :, start:end, :] = y_s

            x_4d = x_4d + temporal_out
            x_frame = rearrange(x_4d, 'b f t d -> (b f) t d')

            if self.spatial_mode == "full":
                x_frame = x_frame + spatial_attn(x_frame)

            else:
                if attn_mask_spatial is None:
                    raise RuntimeError("block spatial mode requires attn_mask_spatial")

                x_frame = x_frame + spatial_attn(
                    x_frame,
                    attn_mask=attn_mask_spatial
                )

            x_4d = rearrange(x_frame, '(b f) t d -> b f t d', f=num_frames)
            x = rearrange(x_4d, 'b f t d -> b (f t) d')

            x = x + ff(x)

        tokens = torch.cat([bos, x], dim=1)
        return self.norm(tokens)

class TENSFormer(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=32,
        num_heads=4,
        num_layers=4,
        ff_mult=4,
        num_tokens=2048,
        num_frames=6
    ):
        super().__init__()
        self.bos_token = nn.Parameter(torch.randn(1, 1, dim))
        self.time_pos_emb = nn.Parameter(torch.randn(1, num_frames, 1, dim))
        self.scale_pos_emb = nn.Parameter(torch.randn(1, 1, num_tokens, dim))
        self.motion_pos_emb = nn.Parameter(torch.randn(1, 1, 1, dim))
        self.temporal_scene_decoder = Decoder(
            dim=dim,
            dim_head=dim_head,
            num_heads=num_heads,
            ff_mult=ff_mult,
            num_layers=num_layers,
            spatial_mode="full",
        )

        self.scale_generation_decoder = Decoder(
            dim=dim,
            dim_head=dim_head,
            num_heads=num_heads,
            ff_mult=ff_mult,
            num_layers=num_layers,
            spatial_mode="block",
        )

    def forward(
        self,
        scene_tokens,
        motion_tokens,
        lengths,
        guidance=None,
        return_guidance_only=False,
    ):
        B, F = scene_tokens.shape[:2]
        device = scene_tokens.device

        scene_tokens = (
            scene_tokens
            + self.time_pos_emb[:, :F]
            + self.scale_pos_emb[:, :, :scene_tokens.size(2)]
        )

        motion_tokens = (
            motion_tokens
            + self.time_pos_emb[:, :F]
            + self.motion_pos_emb
        )

        if not torch.is_tensor(lengths):
            lengths = torch.tensor(lengths, device=device, dtype=torch.long)
        else:
            lengths = lengths.to(device=device, dtype=torch.long)

        ends = torch.cumsum(lengths, dim=0)
        max_cols_per_scale = ends - 1
        max_col_for_row = torch.repeat_interleave(max_cols_per_scale, lengths)

        N = int(max_col_for_row.shape[0])
        col_idx = torch.arange(N, device=device)

        attn_mask_spatial = col_idx.unsqueeze(0) <= max_col_for_row.unsqueeze(1)
        tokens = torch.cat([motion_tokens, scene_tokens], dim=2)

        bos_token = self.bos_token.expand(B, 1, -1)
        tokens = torch.cat(
            [bos_token, rearrange(tokens, 'b f t d -> b (f t) d')],
            dim=1,
        )

        if guidance is None:
            guidance = self.temporal_scene_decoder(
                tokens,
                num_frames=F,
                lengths=lengths,
                attn_mask_spatial=None,
            )

        if return_guidance_only:
            return {
                "guidance": guidance,
            }

        embedding = self.scale_generation_decoder(
            tokens,
            num_frames=F,
            lengths=lengths,
            attn_mask_spatial=attn_mask_spatial,
            guidance=guidance,
        )

        return {
            "full_embedding": embedding,
            "guidance": guidance,
        }