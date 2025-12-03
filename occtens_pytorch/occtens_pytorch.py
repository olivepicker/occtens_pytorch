#wip
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange

from networks.motion_tokenizer import MotionTokenizer
from networks.scene_tokenizer import MultiScaleVQVAE
from networks.tensformer import TENSFormer

class OccTENS(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        num_heads = 8,
        num_layers = 4,
        ff_mult = 4,
        scene_in_channel = 4,
        scene_weight_path = None,
        scene_hidden_channels = 128,
        scene_latent_dim = 128,
        scene_num_codes = 4096,
        scene_scales = (1,5,10,15,20,25),
        scene_enc_kernel_size = [4,3,4,3],
        motion_x_range = (-10, 10),
        motion_y_range = (-10, 10),
        motion_t_range = (-np.pi, np.pi),
        motion_xyt_n_bins = (20, 20, 20)
    ):
        super().__init__()
        self.scene_tokenizer = MultiScaleVQVAE(
            in_channels = scene_in_channel,
            hidden_channels = scene_hidden_channels,
            latent_dim = scene_latent_dim,
            num_codes = scene_num_codes,
            scales = scene_scales,
            enc_kernel_size = scene_enc_kernel_size
        )

        if scene_weight_path is not None:
            w = torch.load(scene_weight_path) #FIXME
            self.scene_tokenizer.load_state_dict(w, strict=True)

        self.motion_tokenizer = MotionTokenizer(
            x_range = motion_x_range,
            y_range = motion_y_range,
            t_range = motion_t_range,
            xyt_n_bins = motion_xyt_n_bins
        )
        self.model = TENSFormer(
            dim = dim,
            dim_head = dim_head,
            num_heads = num_heads,
            num_layers = num_layers,
            ff_mult = ff_mult
        )

        self.scene_tokenizer.eval()
        for p in self.scene_tokenizer.parameters():
            p.requires_grad_(False)

        scene_num_embeddings = self.scene_tokenizer.num_codes
        motion_num_embeddings = self.motion_tokenizer.n_x * self.motion_tokenizer.n_y * self.motion_tokenizer.n_t
        
        self.scene_embedding = nn.Embedding(scene_num_embeddings, dim)
        self.motion_embedding = nn.Embedding(motion_num_embeddings, dim)

    def forward(self, scene, motion):
        device = scene.device
        B, F, C, H, W = scene.size()

        scene = rearrange(scene, 'b f c h w -> (b f) c h w')
        with torch.no_grad():
            _, scene_token_list, _, _ = self.scene_tokenizer(scene)
        scene_ids = torch.cat([rearrange(i, '(b f) h w -> b f (h w)', b=B, f=F) for i in scene_token_list], dim=2)
        scene_tokens = self.scene_embedding(scene_ids)

        motion = rearrange(motion, 'b f c n -> (b f) c n')
        motion_ids = self.motion_tokenizer(motion)
        motion_ids = rearrange(motion_ids, '(b f) t -> b f t', b=B)
        motion_tokens = self.motion_embedding(motion_ids)

        scene_lengths = torch.tensor(
            [s.shape[1] * s.shape[2] for s in scene_token_list],
            device=device,
            dtype=torch.long
        )
        lengths = torch.cat([
            torch.tensor([motion_ids.shape[2]], device=device, dtype=torch.long),
            scene_lengths
        ], dim=0) 

        ret = self.model(
            scene_tokens = scene_tokens,
            motion_tokens = motion_tokens,
            lengths = lengths
        )

        return ret