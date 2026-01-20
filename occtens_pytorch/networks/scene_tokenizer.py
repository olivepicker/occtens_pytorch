import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class VectorQuantizer(nn.Module):
    def __init__(
        self, 
        num_codes: int, 
        code_dim: int, 
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim

        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z):
        B, C, H, W = z.shape
        assert C == self.code_dim, f"code_dim mismatch: {C} != {self.code_dim}"

        # (B, C, H, W) -> (B*H*W, C)
        z_perm = z.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        z_flat = z_perm.view(-1, C)                  # (N, C), N = B*H*W

        z_norm = F.normalize(z_flat, dim=1)
        codebook_norm = F.normalize(self.codebook.weight, dim=1)

        sim = z_norm @ codebook_norm.t()
        encoding_indices = torch.argmax(sim, dim=1)

        z_q_flat = self.codebook(encoding_indices)
        z_q = z_q_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        z_q_st = z + (z_q - z).detach()

        encodings_onehot = F.one_hot(encoding_indices, self.num_codes).float()  # (N, K)
        avg_probs = encodings_onehot.mean(dim=0)  # (K,)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        indices = encoding_indices.view(B, H, W)

        return z_q, indices, perplexity


class Phi(nn.Conv2d):
    def __init__(self, dim, quant_residual=0.5):
        super().__init__(in_channels=dim, out_channels=dim, kernel_size=3, padding=1)
        self.resi_ratio = abs(quant_residual)

    def forward(self, h):
        return h * (1 - self.resi_ratio) + super().forward(h) * self.resi_ratio


class MultiScaleVQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        latent_dim: int = 128,
        num_codes: int = 4096,
        scales = (1,5,10,15,20,25),
        quant_residual = 0.5,
        beta = 0.25,
        num_classes = 18
    ):
        super().__init__()
    
        self.scales = list(scales)
        self.num_codes = num_codes
        self.beta = beta
        self.num_classes = num_classes

        # VQ
        self.vq = VectorQuantizer(num_codes=num_codes, code_dim=latent_dim)

        self.phi = nn.ModuleList([
            Phi(latent_dim, quant_residual=quant_residual) for _ in self.scales
        ])

        self.encoder = Encoder(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            latent_dim=latent_dim, 
        )
        self.decoder = Decoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels, 
            latent_dim=latent_dim, 
        )

        self.pre_quant_conv = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=3//2)
        self.post_quant_conv = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=3//2)

    def encode(self, x):
        stats = {}

        f = self.pre_quant_conv(self.encoder(x))  # (B, D, H_lat, W_lat), latent space
        
        f_no_grad = f.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        B, D, H_lat, W_lat = f.shape

        z_q_list = []
        indices_list = []
        vq_loss_sum = 0.0

        for idx, s in enumerate(self.scales):

            f_rest_nc = F.interpolate(f_rest, size=(s, s), mode="area")

            z_q_s, idx_s, perplex_s = self.vq(f_rest_nc)

            z_q_list.append(z_q_s.view(B, D, -1))   # (B, D, s, s)
            indices_list.append(idx_s)      # (B, s, s)
            
            z = F.interpolate(z_q_s, size=(H_lat, W_lat), mode="bicubic").contiguous()
            f_p = self.phi[idx](z)
            f_hat = f_hat + f_p
            f_rest = f_rest - f_p

            vq_loss_sum += F.mse_loss(f_hat.data, f) * self.beta + F.mse_loss(f_hat, f_no_grad)
            stats[f"perplexity_s{s}"] = perplex_s.detach()
    
        f_hat = (f_hat.data - f_no_grad).add_(f)

        return f_hat, indices_list, stats, vq_loss_sum / len(self.scales)

    @torch.no_grad()
    def decode_from_indices(self, indices_list, out_zyx = True):
        assert isinstance(indices_list, (list, tuple)), "indices_list must be a list/tuple"
        assert len(indices_list) == len(self.scales), f"indices_list length {len(indices_list)} != num_scales {len(self.scales)}"

        device = indices_list[0].device
        dtype = self.vq.codebook.weight.dtype

        B = indices_list[0].shape[0]
        D = self.vq.code_dim if hasattr(self.vq, "code_dim") else self.vq.codebook.embedding_dim

        H_lat = W_lat = self.scales[-1]
        f = torch.zeros((B, D, H_lat, W_lat), device=device, dtype=dtype)

        for idx, idx_s in enumerate(indices_list):
            z = self.vq.codebook(idx_s).permute(0,3,1,2)
            if idx < len(self.scales)-1:
                z = F.interpolate(z, size=(H_lat, W_lat), mode="bicubic").contiguous()
            f += self.phi[idx](z)

        logits = self.decoder(self.post_quant_conv(f))

        if out_zyx:
            logits_3d = rearrange(logits, "b (z c) y x -> b c z y x", c=self.num_classes).contiguous()
            pred_3d = logits_3d.argmax(dim=1)

            return pred_3d

        return logits

    def forward(self, x, mask=None, return_token_only=False):
        B, Z, Y, X = x.size()
        y = x.clone().long()

        # if mask is not None:
        #     y.masked_fill_(~mask.bool(), 255) # Commented out masking for Channel 17 (free) due to training issues. 
        
        x_one_hot = F.one_hot(x, num_classes=self.num_classes)
        x = rearrange(x_one_hot, 'b z y x c -> b (z c) y x').float()

        f_hat, indices_list, stats, vq_loss_sum = self.encode(x)

        if return_token_only:
            return indices_list
            
        x_hat = self.decoder(self.post_quant_conv(f_hat))

        stats['x'] = rearrange(x_one_hot, 'b z y x c -> b c z y x')
        stats['y'] = y
        stats['logits'] = x_hat
        stats['vq_loss_sum'] = vq_loss_sum

        return stats


# Upsample, Downsample class
# https://github.com/FoundationVision/VAR/blob/78b95394fc5896192e3a003e4b295f8ea743c48f/models/basic_vae.py#L22

class Upsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


class Downsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, drop_rate=0.0):
        super().__init__()

        if out_channels == None:
            out_chanenls = in_channels

        if out_channels != in_channels:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut_conv = nn.Identity()
            
        self.block0 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        
    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x0)

        return self.shortcut_conv(x) + x1


class AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)

        q = self.q(x).reshape(b, c, h*w).permute(0, 2, 1)   # (b, hw, c)
        k = self.k(x).reshape(b, c, h*w)                    # (b, c, hw)
        v = self.v(x).reshape(b, c, h*w).permute(0, 2, 1)   # (b, hw, c)

        attn = (q @ k) * (c ** -0.5)                        # (b, hw, hw)
        attn = attn.softmax(dim=-1)

        out = attn @ v                                      # (b, hw, c)
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        out = self.proj(out)

        return x_in + out


class Encoder(nn.Module):
    def __init__(
        self, 
        *,
        in_channels: int, 
        hidden_channels: int,
        latent_dim: int,
        multiple = (1,2,4,8),
        using_mid_attn = False
    ):
        super().__init__()

        self.init_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)

        self.down0 = nn.Sequential(
            ResBlock(hidden_channels * 1, hidden_channels * 1),
            ResBlock(hidden_channels * 1, hidden_channels * 1),
            Downsample2x(hidden_channels * 1)
        )
        
        self.down1 = nn.Sequential(
            ResBlock(hidden_channels * 1, hidden_channels * 2),
            ResBlock(hidden_channels * 2, hidden_channels * 2),
            Downsample2x(hidden_channels * 2)
        )
        self.down2 = nn.Sequential(
            ResBlock(hidden_channels * 2, hidden_channels * 4),
            ResBlock(hidden_channels * 4, hidden_channels * 4),
            Downsample2x(hidden_channels * 4)
        )

        self.down3 = nn.Sequential(
            ResBlock(hidden_channels * 4, hidden_channels * 8),
            ResBlock(hidden_channels * 8, hidden_channels * 8),
        )

        self.mid_block1 = ResBlock(hidden_channels * 8, hidden_channels * 8)
        self.mid_attn   = AttnBlock(hidden_channels * 8) if using_mid_attn else nn.Identity()
        self.mid_block2 = ResBlock(hidden_channels * 8, hidden_channels * 8)

        self.to_latent = nn.Sequential(
            nn.GroupNorm(32, hidden_channels * 8),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels * 8, latent_dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):                
        x0 = self.init_conv(x)
        x1 = self.down0(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.mid_block2(self.mid_attn(self.mid_block1(x4)))

        f  = self.to_latent(x4)

        return f
    

class Decoder(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        latent_dim: int, 
        multiple = (1,2,4,8),
        using_mid_attn = False
    ):
        super().__init__()
        self.init_conv = nn.Conv2d(latent_dim, hidden_channels * 8, kernel_size=3, stride=1, padding=1)

        self.mid_block1 = ResBlock(hidden_channels * 8, hidden_channels * 8)
        self.mid_attn   = AttnBlock(hidden_channels * 8) if using_mid_attn else nn.Identity()
        self.mid_block2 = ResBlock(hidden_channels * 8, hidden_channels * 8)

        self.down0 = nn.Sequential(
            ResBlock(hidden_channels * 8, hidden_channels * 8),
            ResBlock(hidden_channels * 8, hidden_channels * 8),
        )
        
        self.down1 = nn.Sequential(
            ResBlock(hidden_channels * 8, hidden_channels * 4),
            ResBlock(hidden_channels * 4, hidden_channels * 4),
            Upsample2x(hidden_channels * 4)
        )
        self.down2 = nn.Sequential(
            ResBlock(hidden_channels * 4, hidden_channels * 2),
            ResBlock(hidden_channels * 2, hidden_channels * 2),
            Upsample2x(hidden_channels * 2)
        )

        self.down3 = nn.Sequential(
            ResBlock(hidden_channels * 2, hidden_channels * 1),
            ResBlock(hidden_channels * 1, hidden_channels * 1),
            Upsample2x(hidden_channels * 1)
        )
        self.to_latent = nn.Sequential(
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, f):
        x4 = self.init_conv(f)
        x4 = self.mid_block2(self.mid_attn(self.mid_block1(x4)))
        
        x3 = self.down0(x4)
        x2 = self.down1(x3)
        x1 = self.down2(x2)
        x0 = self.down3(x1)
        f = self.to_latent(x0)

        return f