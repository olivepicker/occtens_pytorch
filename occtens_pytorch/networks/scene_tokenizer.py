import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class VectorQuantizer(nn.Module):
    def __init__(
        self, 
        num_codes: int, 
        code_dim: int, 
        beta: float = 0.25
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta

        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z):
        B, C, H, W = z.shape
        assert C == self.code_dim, f"code_dim mismatch: {C} != {self.code_dim}"

        # (B, C, H, W) -> (B*H*W, C)
        z_perm = z.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        z_flat = z_perm.view(-1, C)                  # (N, C), N = B*H*W

        codebook = self.codebook.weight              # (K, C)

        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z·e
        z_sq = (z_flat ** 2).sum(dim=1, keepdim=True)    # (N, 1)
        e_sq = (codebook ** 2).sum(dim=1)                # (K,)
        ze = z_flat @ codebook.t()                       # (N, K)
        distances = z_sq + e_sq - 2 * ze                 # (N, K)

        encoding_indices = torch.argmin(distances, dim=1)  # (N,)
        z_q_flat = self.codebook(encoding_indices)         # (N, C)

        z_q = z_q_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        codebook_loss = F.mse_loss(z_q.detach(), z)
        commitment_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        z_q_st = z + (z_q - z).detach()

        encodings_onehot = F.one_hot(encoding_indices, self.num_codes).float()  # (N, K)
        avg_probs = encodings_onehot.mean(dim=0)  # (K,)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        indices = encoding_indices.view(B, H, W)

        return z_q_st, indices, vq_loss, perplexity

class MultiScaleVQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        latent_dim: int = 128,
        num_codes: int = 4096,
        scales=(1,5,10,15,20,25),
        enc_kernel_size=[4,4,4,3]
    ):
        super().__init__()
    
        self.scales = list(scales)
        self.num_codes = num_codes

        # VQ
        self.vq = VectorQuantizer(num_codes=num_codes, code_dim=latent_dim)

        self.phi_enc = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            for _ in self.scales
        ])

        self.phi_dec = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            for _ in self.scales
        ])

        self.encoder = Encoder(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            latent_dim=latent_dim, 
            kernel_size=enc_kernel_size
        )
        self.decoder = Decoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels, 
            latent_dim=latent_dim, 
            kernel_size=enc_kernel_size[::-1]
        )

    def encode(self, x, return_token_only=False):
        stats = {}

        f = self.encoder(x)  # (B, D, H_lat, W_lat), latent space
        B, D, H_lat, W_lat = f.shape

        z_q_list = []
        indices_list = []
        vq_loss_sum = 0.0

        for idx, s in enumerate(self.scales):
            if s == H_lat and s == W_lat:
                f_hat = f
            else:
                f_hat = F.interpolate(f, size=(s, s), mode="area")

            z_q_s, idx_s, vq_loss_s, perplex_s = self.vq(f_hat)

            z_q_list.append(z_q_s.view(B, D, -1))   # (B, D, s, s)
            indices_list.append(idx_s)      # (B, s, s)
            vq_loss_sum = vq_loss_sum + vq_loss_s
            
            z = self.vq.codebook(idx_s).permute(0,3,1,2)
            z = F.interpolate(z, size=(H_lat, W_lat), mode="bicubic").contiguous()
            f = f - self.phi_enc[idx](z)

            stats[f"perplexity_s{s}"] = perplex_s.detach()

        if return_token_only:
            return torch.cat(z_q_list, dim=2)
        
        return f, z_q_list, indices_list, vq_loss_sum, stats

    def decode(self, f, indices_list):
        B, D, H_lat, W_lat = f.shape
        f = torch.zeros_like(f)

        for idx, idx_s in enumerate(indices_list):
            z = self.vq.codebook(idx_s).permute(0,3,1,2)
            z = F.interpolate(z, size=(H_lat, W_lat), mode="bicubic").contiguous()
            f = f + self.phi_dec[idx](z)

        recon = self.decoder(f)
        return recon

    def forward(self, x):
        B, Z, Y, X = x.size()
        y = x.clone()

        x_one_hot = F.one_hot(x, num_classes=18)
        x = rearrange(x_one_hot, 'b z y x c ->  b (z c) y x').float()
        F_latent, z_q_list, indices_list, vq_loss_sum, stats = self.encode(x)
        x_hat = self.decode(F_latent, indices_list)

        recon_loss = F.binary_cross_entropy_with_logits(x_hat, x)
        #total_loss = recon_loss + vq_loss_sum

        stats['x'] = rearrange(x_one_hot, 'b z y x c -> b c z y x')
        stats['y'] = y
        stats['logits'] = x_hat
        stats["recon_loss"] = recon_loss.detach()
        stats["vq_loss"] = vq_loss_sum.detach()

        return stats
    
class Encoder(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int,
        latent_dim: int,
        kernel_size = [3,4,4,4],
    ):
        super().__init__()
        self.comp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(hidden_channels // 2, hidden_channels, 
                      kernel_size=kernel_size[0], stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, 
                      kernel_size=kernel_size[1], stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, 
                      kernel_size=kernel_size[2], stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, latent_dim, 
                      kernel_size=kernel_size[3], stride=1, padding=1),
        )

    def forward(self, x):
        x = self.comp(x)
        x = self.encoder(x)

        return x
    
class Decoder(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int,
        latent_dim: int,
        kernel_size = [4,4,3,4]
    ):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_channels, 
                      kernel_size=kernel_size[0], stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_channels, hidden_channels,
                               kernel_size=kernel_size[1], stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_channels, hidden_channels,
                               kernel_size=kernel_size[2], stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_channels, hidden_channels  // 2, 
                               kernel_size=kernel_size[3], stride=2, padding=1),
        )
        self.decomp = nn.Conv2d(hidden_channels // 2, in_channels, kernel_size=1)

    def forward(self, x):
        x = self.decoder(x)
        x = self.decomp(x)

        return x