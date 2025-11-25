import torch
import torch.nn as nn

#wip
class MotionTokenizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.x_q = UniformMotionQuantizer()
        self.y_q = UniformMotionQuantizer()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x, y, t):
        prod = self.cartesian_product(x, y, t)
        token = self.embedding(prod)

        return token
        
    def cartesian_product(self, x, y, t):
        v_x = self.x_q(x)
        v_y = self.y_q(y)
        prod = x + y * v_x + t * v_x * v_y

        return prod

class UniformMotionQuantizer:
    def __init__(self, v_min, v_max, num_bins):
        self.v_min = v_min
        self.v_max = v_max
        self.num_bins = num_bins
        self.bin_width = (v_max - v_min) / num_bins

    def __call__(self, v):
        v_clamped = torch.clamp(v, self.v_min, self.v_max - 1e-6)
        indices = torch.floor((v_clamped - self.v_min) / self.bin_width)
        indices = indices.long()  # (B,)
        return indices