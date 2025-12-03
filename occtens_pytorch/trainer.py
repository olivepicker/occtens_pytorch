import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from loss import CustomSceneLoss

#wip
class SceneTokenizerTrainer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.criterion = CustomSceneLoss()
        self.model = model
    
    def train_one_step(self):
        self.model.train()
        pass

    def valid_one_step(self):
        self.model.eval()
        pass

#wip
class OccTENSTrainer(nn.Module):
    def __init__(
        self, 
        model,
        context_frame_point=4,
        ignore_index=-1,
        do_valid=True,
        beta_scene = 0.5,
        beta_motion = 0.5
    ):
        super().__init__()
        self.model = AutoRegressiveWrapper(
            model, 
            context_frame_point=context_frame_point,
            ignore_index=ignore_index
        )

    def train_one_step(self):
        self.model.train()
        batch = None
        

    def valid_one_step(self):
        self.model.eval()
        pass

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

    def forward(self, scene, motion):
        out = self.model(scene=scene, motion=motion)
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

    def generate(self, batch):
        pass