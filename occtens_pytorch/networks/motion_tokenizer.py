import torch
import numpy as np

class MotionTokenizer:
    def __init__(
        self, 
        x_range = (-10, 10),
        y_range = (-10, 10),
        t_range = (-np.pi, np.pi),
        xyt_n_bins = (20, 20, 20)
    ):
        self.n_x, self.n_y, self.n_t = xyt_n_bins

        self.x_q = UniformMotionQuantizer(x_range[0], x_range[1], self.n_x)
        self.y_q = UniformMotionQuantizer(y_range[0], y_range[1], self.n_y)
        self.t_q = UniformMotionQuantizer(t_range[0], t_range[1], self.n_t)

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