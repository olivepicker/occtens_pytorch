#wip
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.motion_tokenizer import MotionTokenizer
from networks.scene_tokenizer import MultiScaleVQVAE
from networks.tensformer import TENSFormer

class OccTENS(nn.Module):
    def __init__(
        self,
        scene_in_channel = None,
        scene_weight_path = '',
        scene_hidden_channels = 128,
        scene_latent_dim = 128,
        scene_num_codes = 4096,
        scene_scales = (1,5,10,15,20,25),
        scene_enc_kernel_size = [4,3,4,3],
        motion_x_range = None,
        motion_y_range = None,
        motion_t_range = None,
        motion_xyt_n_bins = (20, 20, 20)
    ):
        self.scene_tokenizer = MultiScaleVQVAE()
        self.motion_tokenizer = MotionTokenizer()
        self.model = TENSFormer()
        super().__init__()

    def forward(self):
        pass
            
class AutoRegressiveWrapper(nn.Module):
    def __init__(
        self,
        model
    ):
        super().__init__()
        self.model = model
    def forward(self,):
        #self.model()
        pass

    def generate(self, batch):
        pass