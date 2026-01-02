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
        motion_x_range = (-1, 1),
        motion_y_range = (-1, 1),
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
            ).eval()

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
    

class AutoRegressiveWrapper(nn.Module):
    def __init__(
        self,
        model,
        context_frame_point=4,
        ignore_index=-1,
    ):
        super().__init__()
        self.model = model
        self.dim = self.model.dim
        self.vocab_size = self.model.vocab_size
        self.ignore_index = ignore_index

        self.lm_head = nn.Linear(self.dim, self.vocab_size)
        self.context_point = context_frame_point

    def forward(self, scene_token_ids, motions):
        out = self.model(scene_token_ids=scene_token_ids, motions=motions)
        x = out['full_embedding'][:,:-1,:]

        token_ids, frame_idx, token_type = \
            map(lambda t:rearrange(t, 'b f t -> b (f t)'), (out['token_ids'], out['frame_idx'], out['token_type']))

        assert torch.max(frame_idx) >= self.context_point, 'context_point must be lower than num frames.'
        
        is_future = frame_idx >= self.context_point
        is_motion = token_type == 0
        is_scene  = token_type == 1

        scene_mask = is_future & is_scene
        motion_mask = is_future & is_motion

        logits = self.lm_head(x)

        losses = F.cross_entropy(
            input = rearrange(logits, 'b t d -> (b t) d'),
            target = rearrange(token_ids, 'b ft -> (b ft)'),
            reduction = 'none',
            ignore_index = self.ignore_index
        )

        scene_loss = losses[rearrange(scene_mask, 'b d -> (b d)')].mean()
        motion_loss = losses[rearrange(motion_mask, 'b d -> (b d)')].mean()

        out = {
            'losses': losses,
            'scene_loss': scene_loss,
            'motion_loss': motion_loss
        }

        return out

    @torch.no_grad()
    def generate(self, scene_token_ids, motions, max_steps=None, temperature=1.0, top_k=None):
        self.model.eval()
        B, F, T = scene_token_ids.shape
        device = scene_token_ids.device

        scene_tokens = scene_token_ids.clone()
        motion_tokens = motions.clone()

        out = self.model(scene_token_ids=scene_tokens.to(self.device), motions=motion_tokens)

        full_emb = out['full_embedding']
        L = full_emb.size(1) - 1

        token_ids  = rearrange(out['token_ids'],  'b f t -> b (f t)')     # (B, L)
        frame_idx  = rearrange(out['frame_idx'],  'b f t -> b (f t)')     # (B, L)
        token_type = rearrange(out['token_type'], 'b f t -> b (f t)')     # (B, L)

        is_future = frame_idx >= self.context_point
        is_motion = token_type == 0
        is_scene  = token_type == 1

        to_fill = is_future & (token_ids == self.ignore_index)

        b_idx, l_idx = torch.nonzero(to_fill, as_tuple=True)
        num_positions = b_idx.numel()

        if max_steps is not None:
            num_positions = min(num_positions, max_steps)

        for step in range(num_positions):
            b = b_idx[step]
            l = l_idx[step]

            out = self.model(scene_token_ids=scene_tokens, motions=motion_tokens)
            x   = out['full_embedding'][:, :-1, :]
            logits = self.lm_head(x)

            logit_bl = logits[b, l] / temperature

            if top_k is not None:
                values, indices = torch.topk(logit_bl, top_k)
                probs = F.softmax(values, dim=-1)
                next_token = indices[torch.multinomial(probs, 1)]
            else:
                next_token = logit_bl.argmax(dim=-1)

            next_token = next_token.long()

            if is_motion[b, l]:
                f = (l // T).item()
                t = (l %  T).item()
                motion_tokens[b, f, t] = next_token
            elif is_scene[b, l]:
                f = (l // T).item()
                t = (l %  T).item()
                scene_tokens[b, f, t] = next_token
            else:
                continue

        return {
            "scene_token_ids": scene_tokens,
            "motions": motion_tokens,
        }