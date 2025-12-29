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
        use_prepared_token_map = True,
        scene_in_channel = 16,
        scene_weight_path = None,
        scene_hidden_channels = 128,
        scene_latent_dim = 128,
        scene_num_codes = 4096,
        scene_scales = (1, 5, 10, 15, 20, 25),
        scene_enc_kernel_size = (4, 4, 4, 3),
        motion_x_range = (-10, 10),
        motion_y_range = (-10, 10),
        motion_t_range = (-np.pi, np.pi),
        motion_xyt_n_bins = (20, 20, 20)
    ):
        super().__init__()

        if use_prepared_token_map:
            scene_weight_path = None
            self.scene_tokenizer = None
        
        else:
            self.scene_tokenizer = MultiScaleVQVAE(
                in_channels = scene_in_channel,
                hidden_channels = scene_hidden_channels,
                latent_dim = scene_latent_dim,
                num_codes = scene_num_codes,
                scales = scene_scales,
                enc_kernel_size = scene_enc_kernel_size
            )
            self.scene_tokenizer.eval()

        if scene_weight_path is not None:
            w = torch.load(scene_weight_path) #FIXME
            self.scene_tokenizer.load_state_dict(w, strict=True)

            for p in self.scene_tokenizer.parameters():
                p.requires_grad_(False)

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

        self.dim = dim
        self.motion_vocab_size = np.prod(motion_xyt_n_bins)
        self.vocab_size = scene_num_codes + self.motion_vocab_size
        self.scene_token_embedding = nn.Embedding(self.vocab_size, dim)
        self.motion_token_embedding = nn.Embedding(self.vocab_size, dim)
        
    def forward(self, scene_token_ids, motions):
        device = scene_token_ids.device
        
        #TODO without pre-generated token maps
        # B, F, C, H, W = scene.size()
        
        # scene = rearrange(scene, 'b f c h w -> (b f) c h w')
        # with torch.no_grad():
        #     _, scene_token_list, _, _ = self.scene_tokenizer(scene)
        # scene_ids = torch.cat([rearrange(i, '(b f) h w -> b f (h w)', b=B, f=F) for i in scene_token_list], dim=2)
        # scene_ids += torch.tensor(self.motion_vocab_size)
        # scene_tokens = self.token_embedding(scene_ids)

        
        B, F, T = scene_token_ids.size()
        scene_tokens = self.scene_token_embedding(scene_token_ids)
        motion_ids = self.motion_tokenizer(motions)[:,:,None]
        motion_tokens = self.motion_token_embedding(motion_ids)#[:,:,None,:]

        scene_length = torch.tensor(
            [scene_token_ids.shape[2]],
            device=device,
            dtype=torch.long
        )

        motion_length = motion_tokens.shape[2]

        lengths = torch.cat([
            torch.tensor([motion_length], device=device, dtype=torch.long),
            scene_length
        ], dim=0) 

        embedding = self.model(
            scene_tokens = scene_tokens,
            motion_tokens = motion_tokens,
            lengths = lengths
        ) # (batch, n_frame, token, dim)

        token_emb = embedding[:, 1:, :]
        token_length = int(lengths.sum().item())
        token_type = torch.zeros((B, F, token_length), device=device, dtype=torch.long)
        token_type[:, :, motion_length:] = 1
        
        out = {
            'full_embedding': embedding,
            'token_embedding': token_emb,
            'token_ids': torch.cat([motion_ids, scene_token_ids], dim=2),
            'scene_ids': scene_token_ids,
            'motion_ids':motion_ids,
            'token_type':token_type,
            'frame_idx': torch.arange(F, device=device).view(1, F, 1).expand(B, F, token_length),
        }

        return out