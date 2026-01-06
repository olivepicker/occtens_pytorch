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

        return z_q, encoding_indices, perplexity

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
        enc_kernel_size = [4,4,4,3],
        quant_residual = 0.5,
        beta = 0.25,
    ):
        super().__init__()
    
        self.scales = list(scales)
        self.num_codes = num_codes
        self.beta = beta

        # VQ
        self.vq = VectorQuantizer(num_codes=num_codes, code_dim=latent_dim)

        self.phi_enc = nn.ModuleList([
            Phi(latent_dim, quant_residual=quant_residual) for _ in self.scales
        ])

        self.phi_dec = nn.ModuleList([
            Phi(latent_dim, quant_residual=quant_residual) for _ in self.scales
        ])

        self.encoder = Encoder(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            latent_dim=latent_dim, 
            #kernel_size=enc_kernel_size
        )
        self.decoder = Decoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels, 
            latent_dim=latent_dim, 
            #kernel_size=enc_kernel_size[::-1]
        )

        self.pre_quant_conv = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=3//2)
        self.post_quant_conv = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=3//2)

    def encode(self, x, return_token_only=False):
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
            f_p = self.phi_enc[idx](z)
            f_hat = f_hat + f_p
            f_rest = f_rest - f_p

            vq_loss_sum += F.mse_loss(f_hat.data, f) * self.beta + F.mse_loss(f_hat, f_no_grad)
            stats[f"perplexity_s{s}"] = perplex_s.detach()

        if return_token_only:
            return torch.cat(z_q_list, dim=2)

        f_hat = (f_hat.data - f_no_grad).add_(f)

        return f_hat, indices_list, stats, vq_loss_sum / len(self.scales)

    # def decode(self, f_hat):
    #     return self.decoder(f_hat)

    def forward(self, x, mask=None):
        B, Z, Y, X = x.size()
        y = x.clone().long()
        y.masked_fill_(~mask.bool(), 255)
        #x.masked_fill_(~mask.bool(), 17)
        
        x_one_hot = F.one_hot(x, num_classes=18)
        #x_one_hot = F.one_hot(x_clamped, num_classes=18)
        #x_one_hot = x_one_hot * valid.unsqueeze(-1)
        x = rearrange(x_one_hot, 'b z y x c -> b (z c) y x').float()

        B, C, H, W = x.size()
        rem = H % 2**4
        if rem != 0:
            x = F.pad(x, (0, rem, 0, rem))

        f_hat, indices_list, stats, vq_loss_sum = self.encode(x)
        x_hat = self.decoder(self.post_quant_conv(f_hat))[...,:H,:W]

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
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
    
    def forward(self, x):
        return self.conv(F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0))


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


class Encoder(nn.Module):
    def __init__(
        self, 
        *,
        in_channels: int, 
        hidden_channels: int,
        latent_dim: int,
        multiple = (1,2,4,8)
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
            Downsample2x(hidden_channels * 8)
        )
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
        f  = self.to_latent(x4)

        return f
    

class Decoder(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        latent_dim: int, 
        multiple = (1,2,4,8)
    ):
        super().__init__()
        self.init_conv = nn.Conv2d(latent_dim, hidden_channels * 8, kernel_size=3, stride=1, padding=1)

        self.down0 = nn.Sequential(
            ResBlock(hidden_channels * 8, hidden_channels * 8),
            ResBlock(hidden_channels * 8, hidden_channels * 8),
            Upsample2x(hidden_channels * 8)
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
        x3 = self.down0(x4)
        x2 = self.down1(x3)
        x1 = self.down2(x2)
        x0 = self.down3(x1)
        f = self.to_latent(x0)

        return f