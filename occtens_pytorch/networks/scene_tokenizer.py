import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, code_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta

        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z):
        """
        z: (B, C, H, W)
        """
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


class Encoder(nn.Module):
    def __init__(self, backbone):
        self.backbone = backbone
    
    def forward_features(self, x):
        return x
    

class Encoder(nn.Module):
    def __init__(self, backbone):
        self.backbone = backbone
    
    def forward_features(self, x):
        return x
    