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
        num_frames = 10,
        ff_mult = 4,
        use_prepared_token_map = True,
        scene_in_channel = 16,
        scene_weight_path = None,
        scene_hidden_channels = 128,
        scene_latent_dim = 128,
        scene_num_codes = 4096,
        scene_scales = (1, 5, 10, 15, 20, 25),
        motion_x_range = (-1, 1),
        motion_y_range = (-1, 1),
        motion_t_range = (-np.pi, np.pi),
        motion_xyt_n_bins = (20, 20, 20),
        ignore_index = -1
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

        num_tokens = np.sum([i*i for i in scene_scales])
        self.model = TENSFormer(
            dim = dim,
            dim_head = dim_head,
            num_heads = num_heads,
            num_layers = num_layers,
            ff_mult = ff_mult,
            num_tokens = num_tokens,
            num_frames = num_frames
        )

        self.dim = dim

        self.scene_vocab_size = scene_num_codes
        self.scene_token_embedding = nn.Embedding(self.scene_vocab_size, dim)
        self.scene_mask_token_id = self.scene_vocab_size

        self.motion_vocab_size = np.prod(motion_xyt_n_bins)
        self.motion_token_embedding = nn.Embedding(self.motion_vocab_size, dim)
        self.motion_mask_token_id = self.motion_vocab_size

        self.ignore_index = ignore_index
        
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
    
        scene_input_ids = scene_token_ids.clone()
        scene_unknown = scene_input_ids == self.ignore_index
        scene_input_ids[scene_unknown] = self.scene_mask_token_id

        scene_tokens = self.scene_token_embedding(scene_input_ids)

        motion_unknown = (motions == self.ignore_index).any(dim=-1)  # (B, F)
        safe_motions = motions.clone()
        safe_motions[motion_unknown] = 0.0
        motion_ids = self.motion_tokenizer(safe_motions)[:, :, None]  # (B, F, 1)
        motion_ids = motion_ids.clone()
        motion_ids[motion_unknown[:, :, None]] = self.motion_mask_token_id

        motion_tokens = self.motion_token_embedding(motion_ids)

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
        
        motion_target_ids = self.motion_tokenizer(safe_motions)[:, :, None]
        motion_target_ids = motion_target_ids.clone()
        motion_target_ids[motion_unknown[:, :, None]] = self.ignore_index


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
    def generate(
        self,
        past_scene_tokens,
        past_motions,
        total_frames,
        context_point=None,
        future_motions=None,
        max_steps=None,
        temperature=1.0,
        top_k=None,
    ):
        self.model.eval()

        if context_point is None:
            context_point = past_scene_tokens.size(1)

        B, C, T_scene = past_scene_tokens.shape
        _, C_m, motion_dim = past_motions.shape

        assert C == C_m
        assert motion_dim == 3

        device = past_scene_tokens.device

        scene_tokens = torch.full(
            (B, total_frames, T_scene),
            fill_value=self.ignore_index,
            device=device,
            dtype=torch.long,
        )
        scene_tokens[:, :context_point, :] = past_scene_tokens

        motion_tokens = torch.full(
            (B, total_frames, 3),
            fill_value=self.ignore_index,
            device=device,
            dtype=torch.long,
        )
        motion_tokens[:, :context_point, :] = past_motions

        if future_motions is not None:
            motion_tokens[:, context_point:, :] = future_motions
            generate_motion = False
        else:
            generate_motion = True

        return self.generate_joint(
            scene_token_ids=scene_tokens,
            motions=motion_tokens,
            context_point=context_point,
            generate_motion=generate_motion,
            max_steps=max_steps,
            temperature=temperature,
            top_k=top_k,
        )
    

    @torch.no_grad()
    def generate_joint(
        self,
        scene_token_ids,
        motions,
        context_point,
        generate_motion=True,
        max_steps=None,
        temperature=1.0,
        top_k=None,
    ):
        B, F_total, T_scene = scene_token_ids.shape
        T_frame = T_scene + 1

        scene_tokens = scene_token_ids.clone()
        motion_tokens = motions.clone()

        out = self.model(scene_token_ids=scene_tokens, motions=motion_tokens)

        token_ids = rearrange(out["token_ids"], "b f t -> b (f t)")
        frame_idx = rearrange(out["frame_idx"], "b f t -> b (f t)")
        token_type = rearrange(out["token_type"], "b f t -> b (f t)")

        is_future = frame_idx >= context_point
        is_motion = token_type == 0
        is_scene = token_type == 1

        if generate_motion:
            to_fill = is_future & (token_ids == self.ignore_index)
        else:
            to_fill = is_future & is_scene & (token_ids == self.ignore_index)

        b_idx, l_idx = torch.nonzero(to_fill, as_tuple=True)

        if max_steps is not None:
            n_steps = min(b_idx.numel(), max_steps)
        else:
            n_steps = b_idx.numel()

        for step in range(n_steps):
            b = b_idx[step]
            l = l_idx[step]

            out = self.model(scene_token_ids=scene_tokens, motions=motion_tokens)
            x = out["full_embedding"][:, :-1, :]
            logits = self.lm_head(x)

            logit_bl = logits[b, l] / temperature

            if top_k is not None:
                k = min(top_k, logit_bl.size(-1))
                values, indices = torch.topk(logit_bl, k)
                probs = torch.softmax(values, dim=-1)
                sample_idx = torch.multinomial(probs, 1).item()
                next_token = indices[sample_idx]
            else:
                next_token = logit_bl.argmax(dim=-1)

            next_token = next_token.long()

            f = (l // T_frame).item()
            local_idx = (l % T_frame).item()

            if local_idx == 0:
                if not generate_motion:
                    continue

                motion_components = self.motion_tokenizer.decode_token(next_token)
                motion_tokens[b, f, :] = motion_components.to(
                    device=motion_tokens.device,
                    dtype=motion_tokens.dtype,
                )

            else:
                scene_t = local_idx - 1
                scene_tokens[b, f, scene_t] = next_token

        return {
            "scene_token_ids": scene_tokens,
            "motions": motion_tokens,
        }