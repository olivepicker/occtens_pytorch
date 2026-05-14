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

        self.scene_scales = scene_scales
        self.scene_scale_lengths = [s * s for s in scene_scales]

        self.scene_vocab_size = scene_num_codes
        self.scene_token_embedding = nn.Embedding(self.scene_vocab_size + 1, dim)
        self.scene_mask_token_id = self.scene_vocab_size

        self.motion_vocab_size = np.prod(motion_xyt_n_bins)
        self.motion_token_embedding = nn.Embedding(self.motion_vocab_size + 1, dim)
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
        motion_length = motion_tokens.shape[2]

        scene_scale_lengths = torch.tensor(
            self.scene_scale_lengths,
            device=device,
            dtype=torch.long,
        )

        if int(scene_scale_lengths.sum().item()) != scene_token_ids.shape[2]:
            raise ValueError(
                f"sum(scene_scale_lengths)={scene_scale_lengths.sum().item()} "
                f"must equal T_scene={scene_token_ids.shape[2]}"
            )

        lengths = torch.cat([
            torch.tensor([motion_length], device=device, dtype=torch.long),
            scene_scale_lengths,
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
        self.motion_vocab_size = self.model.motion_vocab_size
        self.scene_vocab_size = self.model.scene_vocab_size
        self.scale_lengths = self.model.scene_scale_lengths
        
        self.ignore_index = ignore_index
        self.context_point = context_frame_point

        self.scene_lm_head = nn.Linear(self.dim, self.scene_vocab_size)
        self.motion_lm_head = nn.Linear(self.dim, self.motion_vocab_size)

    def forward(self, scene_token_ids, motions):
        out = self.model(scene_token_ids=scene_token_ids, motions=motions)

        x = out["full_embedding"][:, :-1, :]

        token_ids, frame_idx, token_type = map(
            lambda t: rearrange(t, "b f t -> b (f t)"),
            (out["token_ids"], out["frame_idx"], out["token_type"]),
        )

        if x.size(1) != token_ids.size(1):
            raise RuntimeError(
                f"x/token_ids length mismatch: x={x.size(1)}, token_ids={token_ids.size(1)}"
            )

        assert torch.max(frame_idx) >= self.context_point, (
            "context_point must be lower than num frames."
        )

        is_future = frame_idx >= self.context_point
        is_motion = token_type == 0
        is_scene = token_type == 1
        is_valid = token_ids != self.ignore_index

        scene_mask = is_future & is_scene & is_valid
        motion_mask = is_future & is_motion & is_valid

        if scene_mask.any():
            scene_x = x[scene_mask]              # (N_scene, D)
            scene_target = token_ids[scene_mask] # (N_scene,)

            scene_logits = self.scene_lm_head(scene_x)

            scene_loss = F.cross_entropy(
                scene_logits,
                scene_target,
                reduction="mean",
            )
        else:
            scene_loss = x.sum() * 0.0

        if motion_mask.any():
            motion_x = x[motion_mask]              # (N_motion, D)
            motion_target = token_ids[motion_mask] # (N_motion,)

            motion_logits = self.motion_lm_head(motion_x)

            motion_loss = F.cross_entropy(
                motion_logits,
                motion_target,
                reduction="mean",
            )
        else:
            motion_loss = x.sum() * 0.0

        loss = scene_loss + motion_loss

        return {
            "loss": loss,
            "scene_loss": scene_loss,
            "motion_loss": motion_loss,
        }

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
            fill_value=float(self.ignore_index),
            device=device,
            dtype=past_motions.dtype,
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
        scale_lengths=None,
        max_steps=None,
        temperature=1.0,
        top_k=None,
    ):
        self.model.eval()

        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")

        B, F_total, T_scene = scene_token_ids.shape
        T_frame = T_scene + 1

        if scale_lengths is None:
            scale_lengths = self.scale_lengths

        if sum(scale_lengths) != T_scene:
            raise ValueError(
                f"sum(scale_lengths) must equal T_scene. "
                f"got sum(scale_lengths)={sum(scale_lengths)}, T_scene={T_scene}"
            )

        scene_tokens = scene_token_ids.clone()
        motion_tokens = motions.clone()

        def sample_logits(logits):
            logits = logits / temperature

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                values, indices = torch.topk(logits, k, dim=-1)
                probs = torch.softmax(values, dim=-1)

                flat_probs = probs.reshape(-1, k)
                sample_idx = torch.multinomial(flat_probs, 1)

                sample_idx = sample_idx.reshape(*logits.shape[:-1], 1)
                next_tokens = torch.gather(indices, -1, sample_idx).squeeze(-1)
                return next_tokens.long()

            return logits.argmax(dim=-1).long()

        generated_steps = 0

        for f in range(context_point, F_total):
            if generate_motion:
                if max_steps is not None and generated_steps >= max_steps:
                    break

                motion_is_unknown = (motion_tokens[:, f, :] == self.ignore_index).any(dim=-1)

                if motion_is_unknown.any():
                    out = self.model(
                        scene_token_ids=scene_tokens,
                        motions=motion_tokens,
                    )
                    x = out["full_embedding"][:, :-1, :]

                    l_motion = f * T_frame  # local_idx == 0
                    h = x[motion_is_unknown, l_motion, :]  # (B_active, D)

                    logits = self.motion_lm_head(h)  # (B_active, motion_vocab)
                    next_motion_tokens = sample_logits(logits)  # (B_active,)

                    motion_components = self.model.motion_tokenizer.decode_token(
                        next_motion_tokens,
                        return_continuous=True,
                    )

                    motion_tokens[motion_is_unknown, f, :] = motion_components.to(
                        device=motion_tokens.device,
                        dtype=motion_tokens.dtype,
                    )

                    generated_steps += 1

            start = 0

            for scale_len in scale_lengths:
                if max_steps is not None and generated_steps >= max_steps:
                    break

                end = start + scale_len
                scale_unknown = scene_tokens[:, f, start:end] == self.ignore_index  # (B, scale_len)

                if not scale_unknown.any():
                    start = end
                    continue

                out = self.model(
                    scene_token_ids=scene_tokens,
                    motions=motion_tokens,
                )
                x = out["full_embedding"][:, :-1, :]

                l_start = f * T_frame + 1 + start
                l_end = f * T_frame + 1 + end

                h = x[:, l_start:l_end, :]  # (B, scale_len, D)

                logits = self.scene_lm_head(h)  # (B, scale_len, scene_vocab)
                next_scene_tokens = sample_logits(logits)  # (B, scale_len)

                current = scene_tokens[:, f, start:end]
                current[scale_unknown] = next_scene_tokens[scale_unknown]
                scene_tokens[:, f, start:end] = current

                generated_steps += 1
                start = end

        return {
            "scene_token_ids": scene_tokens,
            "motions": motion_tokens,
        }