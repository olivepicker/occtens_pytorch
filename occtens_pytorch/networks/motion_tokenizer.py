import torch
import numpy as np

class MotionTokenizer:
    def __init__(
        self, 
        x_range = (-1, 1),
        y_range = (-1, 1),
        t_range = (-np.pi, np.pi),
        xyt_n_bins = (20, 20, 20)
    ):
        self.n_x, self.n_y, self.n_t = xyt_n_bins

        self.x_q = UniformMotionQuantizer(x_range[0], x_range[1], self.n_x)
        self.y_q = UniformMotionQuantizer(y_range[0], y_range[1], self.n_y)
        self.t_q = UniformMotionQuantizer(t_range[0], t_range[1], self.n_t)

    @property
    def vocab_size(self):
        return self.n_x * self.n_y * self.n_t
        
    def __call__(self, xyt):
        x, y, t = xyt[:,:,0], xyt[:,:,1], xyt[:,:,2]
        token = self.cartesian_product(x, y, t)

        return token
        
    def cartesian_product(self, x, y, t):
        i_x = self.x_q(x)
        i_y = self.y_q(y)
        i_t = self.t_q(t)

        prod = i_x + (i_y * self.n_x) + (i_t * self.n_x * self.n_y) # x + y × Vx + θ × Vx × Vy

        return prod
    
    def decode_token(self, token, return_continuous=True):
        token = token.long()

        if torch.any(token < 0) or torch.any(token >= self.vocab_size):
            raise ValueError(
                f"Motion token out of range. "
                f"Expected [0, {self.vocab_size - 1}], got "
                f"min={token.min().item()}, max={token.max().item()}"
            )

        i_t = token // (self.n_x * self.n_y)
        rem = token % (self.n_x * self.n_y)

        i_y = rem // self.n_x
        i_x = rem % self.n_x

        indices = torch.stack([i_x, i_y, i_t], dim=-1)

        if not return_continuous:
            return indices

        x = self.x_q.decode_index(i_x)
        y = self.y_q.decode_index(i_y)
        t = self.t_q.decode_index(i_t)

        return torch.stack([x, y, t], dim=-1)


class UniformMotionQuantizer:
    def __init__(self, v_min, v_max, num_bins):
        self.v_min = v_min
        self.v_max = v_max
        self.num_bins = num_bins
        self.bin_width = (v_max - v_min) / num_bins

    def __call__(self, v):
        v_clamped = torch.clamp(v, self.v_min, self.v_max - 1e-6)
        indices = torch.floor((v_clamped - self.v_min) / self.bin_width)

        return indices.long()

    def decode_index(self, indices):
        indices = indices.to(torch.float32)

        return self.v_min + (indices + 0.5) * self.bin_width