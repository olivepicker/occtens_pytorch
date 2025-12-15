import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from torch.utils.data import DataLoader
from einops import rearrange

from loss import CustomSceneLoss


class SceneTokenizerTrainer(nn.Module):
    def __init__(
        self, 
        model,
        optimizer,
        train_ds,
        valid_ds,
        device='cuda',
        autocast_enabled=False,
        autocast_device_type='cuda',
        autocast_dtype=torch.float16,
        batch_size=4,
        num_workers=4,
        lambda_ce=10.0,
        lambda_lovasz=1.0,
        lambda_geoscal=0.3,
        lambda_semscal=0.5,
        ignore_index=255,
        lambda_recon=1.0, 
        lambda_vq=1.0,
        save_path = 'scene_output/'
    ):
        super().__init__()

        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

        self.train_ds = train_ds
        self.valid_ds = valid_ds

        self.train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.valid_dl = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        self.criterion = CustomSceneLoss(
            lambda_ce=lambda_ce,
            lambda_geoscal=lambda_geoscal,
            lambda_lovasz=lambda_lovasz,
            lambda_semscal=lambda_semscal,
            ignore_index=ignore_index,
            num_classes=18,
        )

        self.lambda_rec = lambda_recon
        self.lambda_vq = lambda_vq
        self.best_val_loss = float('inf')

        self.autocast_config = {
            'device_type':autocast_device_type,
            'dtype':autocast_dtype,
            'enabled':autocast_enabled
        }

        self.save_path = save_path
        if os.path.exists(self.save_path)==False:
            os.makedirs(self.save_path)

    def train_one_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        x = batch["semantic"].to(self.device)

        with torch.autocast(**self.autocast_config):
            out = self.model(x)
            logits = out["logits"]
            #vq_loss = out["vq_loss"]
            
            loss_dict = self.criterion(logits, out['y'])
            rec_loss = loss_dict["loss"]
        
        total_loss = self.lambda_rec * rec_loss #+ self.lambda_vq * vq_loss
        total_loss.backward()
        self.optimizer.step()

        return {
            "loss_total": total_loss.detach(),
            "loss_rec": rec_loss.detach(),
            #"loss_vq": vq_loss.detach(),
        }

    def valid_one_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            x = batch["semantic"].to(self.device)

            with torch.autocast(**self.autocast_config):
                out = self.model(x)
                logits = out["logits"]
                #vq_loss = out["vq_loss"]
                
                loss_dict = self.criterion(logits, out['y'])
                rec_loss = loss_dict["loss"]
            
            total_loss = self.lambda_rec * rec_loss #+ self.lambda_vq * vq_loss

        return {
            "loss_total": total_loss,
            "loss_rec": rec_loss,
            #"loss_vq": vq_loss,
        }
    
    def train(self, num_epochs, log_interval=50, val_interval=1):
        for epoch in range(num_epochs):
            self.model.train()
            train_loss_sum = 0.0
            for step, batch in enumerate(self.train_dl):
                log = self.train_one_step(batch)
                train_loss_sum += log["loss_total"].item()

                if (step + 1) % log_interval == 0:
                    avg = train_loss_sum / (step + 1)
                    print(f"[Epoch {epoch+1} | Step {step+1}] "
                          f"train_loss={avg:.4f}")

            if (epoch + 1) % val_interval == 0:
                self.model.eval()
                val_loss_sum = 0.0
                with torch.no_grad():
                    for batch in self.valid_dl:
                        log = self.valid_one_step(batch)
                        val_loss_sum += log["loss_total"].item()
                val_avg = val_loss_sum / max(1, len(self.valid_dl))
                print(f"[Epoch {epoch+1}] val_loss={val_avg:.4f}")

                if val_avg < self.best_val_loss:
                    print(f"Validation loss improved from {self.best_val_loss:.4f} to {val_avg:.4f}. Saving best model...")
                    self.best_val_loss = val_avg
                    
                    save_path = os.path.join(self.save_path, "best_model.pth")
                    torch.save(self.model.state_dict(), save_path)
                
                last_path = os.path.join(self.save_path, "last_model.pth")
                torch.save(self.model.state_dict(), last_path)

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
        self.beta_scene = beta_scene
        self.beta_motion = beta_motion

    def train_one_step(self):
        self.model.train()
        batch = None
        out = self.model(scene=batch['scene'], motion=batch['motion'])
        loss = out['scene_loss'] * self.beta_scene + out['motion_loss'] * self.beta_motion
        loss.backward()

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