import torch
import torch.nn as nn
import torch.nn.functional as F

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
        latent_dim: int = 64,
        num_codes: int = 1024,
        scales=(1,5,10,15,20,25),
        enc_kernel_size=[4,3,4,3]
    ):
        super().__init__()

        self.scales = list(scales)
        self.n_scales = len(scales)

        # VQ
        self.vq = VectorQuantizer(num_codes=num_codes, code_dim=latent_dim)
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

    def encode_multi_scale(self, x):
        stats = {}

        F_latent = self.encoder(x)  # (B, D, H_lat, W_lat)
        B, D, H_lat, W_lat = F_latent.shape

        z_q_list = []
        indices_list = []
        vq_loss_sum = 0.0

        for s in self.scales:
            # Downsample
            if s == H_lat and s == W_lat:
                F_s = F_latent
            else:
                F_s = F.interpolate(F_latent, size=(s, s),
                                    mode="bilinear", align_corners=False)

            # VQ
            z_q_s, idx_s, vq_loss_s, perplex_s = self.vq(F_s)

            z_q_list.append(z_q_s)          # (B, D, s, s)
            indices_list.append(idx_s)      # (B, s, s)
            vq_loss_sum = vq_loss_sum + vq_loss_s

            stats[f"perplexity_s{s}"] = perplex_s.detach()

        return F_latent, z_q_list, indices_list, vq_loss_sum, stats

    def decode_multi_scale(self, F_latent, z_q_list):
        B, D, H_lat, W_lat = F_latent.shape
        up_list = []

        for i, (s, z_q_s) in enumerate(zip(self.scales, z_q_list)):
            if s == H_lat and s == W_lat:
                up = z_q_s
            else:
                up = F.interpolate(z_q_s, size=(H_lat, W_lat),
                                   mode="bilinear", align_corners=False)

            up = self.phi_dec[i](up)
            up_list.append(up)

        F_hat = torch.stack(up_list, dim=0).sum(dim=0)  # (B, D, H_lat, W_lat)
        return F_hat

    def forward(self, x):
        F_latent, z_q_list, indices_list, vq_loss_sum, stats = self.encode_multi_scale(x)
        F_hat = self.decode_multi_scale(F_latent, z_q_list)
        x_hat = self.decoder(F_hat)

        recon_loss = F.binary_cross_entropy_with_logits(x_hat, x)
        total_loss = recon_loss + vq_loss_sum

        stats["recon_loss"] = recon_loss.detach()
        stats["vq_loss"] = vq_loss_sum.detach()

        return x_hat, indices_list, total_loss, stats
    
class Encoder(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int,
        latent_dim: int,
        kernel_size = [4,3,4,3],
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 2, 
                      kernel_size=kernel_size[0], stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels // 2, hidden_channels, 
                      kernel_size=kernel_size[1], stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, 
                      kernel_size=kernel_size[2], stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, latent_dim, 
                      kernel_size=kernel_size[3], stride=1, padding=1),
        )

    def forward(self, x):
        return self.encoder(x)
    
class Decoder(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int,
        latent_dim: int,
        kernel_size = [3,4,3,4]
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_channels, 
                      kernel_size=kernel_size[0], padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_channels, hidden_channels,
                               kernel_size=kernel_size[1], stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2,
                               kernel_size=kernel_size[2], stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_channels // 2, in_channels, 
                               kernel_size=kernel_size[3], stride=2, padding=1),
        )

    def forward(self, x):
        return self.decoder(x)