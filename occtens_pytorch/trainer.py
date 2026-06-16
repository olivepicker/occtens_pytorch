import torch
import torch.nn as nn
import os
import numpy as np

from torch.utils.data import DataLoader, ConcatDataset
from transformers import get_cosine_schedule_with_warmup
from einops import rearrange
from tqdm.auto import tqdm

from loss import CustomSceneLoss
from occtens_pytorch import AutoRegressiveWrapper


class SceneTokenizerTrainer(nn.Module):
    def __init__(
        self, 
        model,
        num_epochs,
        optimizer,
        train_ds,
        valid_ds,
        use_scheduler=True,
        device='cuda',
        autocast_enabled=False,
        autocast_device_type='cuda',
        autocast_dtype=torch.float16,
        batch_size=4,
        num_workers=4,
        num_classes=18,
        free_class_index=17,
        lambda_ce=10.0,
        lambda_lovasz=1.0,
        lambda_geoscal=0.3,
        lambda_semscal=0.5,
        ignore_index=255,
        lambda_recon=1.0, 
        lambda_vq=1.0,
        save_path='scene_output/',
        save_token=False
    ):
        super().__init__()

        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs

        self.save_token = save_token
        if self.save_token:
            print('train_ds, valid_ds concatenated for save token')
            valid_ds = ConcatDataset([train_ds, valid_ds])

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
            num_classes=num_classes,
            free_class_index=free_class_index
        )

        self.lambda_rec = lambda_recon
        self.lambda_vq = lambda_vq
        self.best_val_loss = float('inf')

        self.autocast_config = {
            'device_type':autocast_device_type,
            'dtype':autocast_dtype,
            'enabled':autocast_enabled
        }
        self.scaler = torch.amp.GradScaler(enabled=autocast_enabled)

        num_training_steps = num_epochs * len(self.train_dl)
        num_warmup_steps = int(num_training_steps * 0.05)

        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer = self.optimizer,
            num_warmup_steps = num_warmup_steps,
            num_training_steps = num_training_steps
        ) if use_scheduler else None

        self.save_path = save_path
        if os.path.exists(self.save_path)==False:
            os.makedirs(self.save_path)

    def train_one_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        x = batch["semantic"].to(self.device)
        mask = batch["mask"].to(self.device)

        with torch.autocast(**self.autocast_config):
            out = self.model(x, mask)
            logits = out["logits"]
            
            loss_dict = self.criterion(logits, out['y'])
            rec_loss = loss_dict["loss"]
            vq_loss = out['vq_loss_sum']
        
        total_loss = rec_loss + vq_loss
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if hasattr(self, 'scheduler') and self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss_total": total_loss.detach(),
        }

    def valid_one_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            x = batch["semantic"].to(self.device)
            mask = batch["mask"].to(self.device)

            with torch.autocast(**self.autocast_config):
                out = self.model(x, mask)
                logits = out["logits"]
                
                loss_dict = self.criterion(logits, out['y'])
                rec_loss = loss_dict["loss"]
                vq_loss = out['vq_loss_sum']
            
            total_loss = rec_loss + vq_loss

        return {
            "loss_total": total_loss.detach(),
        }
    
    def train(self, log_interval=50, val_interval=1):
        
        for epoch in range(self.num_epochs):
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

    def save_token_all(self):
        assert self.save_token == True, 'save_token should be True when save token'
        self.model.eval()
        token_save_path = os.path.join(f'{self.save_path}', 'tokens')

        if os.path.exists(token_save_path) == False:
            os.makedirs(token_save_path)

        for idx, batch in tqdm(enumerate(self.valid_dl), total=len(self.valid_dl)):
            batch['semantic'] = batch['semantic'].to(self.device)
            batch['mask'] = batch['mask'].to(self.device)
            B = batch['semantic'].size(0)

            with torch.no_grad():
                tokens = self.model(batch['semantic'], return_token_only=True)
                tokens = [rearrange(i, 'b x y -> b (x y)') for i in tokens]
                tokens = torch.cat(tokens, dim=1).detach().cpu().numpy()

                for b in range(B):
                    scene_num = batch['scene_num'][b]
                    scene_id = batch['scene_id'][b]
                    np.save(os.path.join(token_save_path, f'{scene_num}_{scene_id}.npy'), tokens[b])


class OccTENSTrainer(nn.Module):
    def __init__(
        self,
        num_epochs,
        model,
        optimizer,
        train_ds,
        valid_ds,
        lr=1e-4,
        device='cuda',
        use_reduced_scale=False,
        autocast_enabled=False,
        autocast_device_type='cuda',
        autocast_dtype=torch.float16,
        batch_size=4,
        num_workers=4,
        context_frame_point=4,
        ignore_index=-1,
        beta_scene=1.,
        beta_motion=1.,
        save_path='occtens_output/'
    ):
        super().__init__()
        self.model = AutoRegressiveWrapper(
            model, 
            context_frame_point=context_frame_point,
            ignore_index=ignore_index,
            use_reduced_scale=use_reduced_scale
        ).to(device)

        self.num_epochs = num_epochs
        self.optimizer = optimizer(lr=lr, params=self.model.parameters())
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

        self.autocast_config = {
            'device_type':autocast_device_type,
            'dtype':autocast_dtype,
            'enabled':autocast_enabled
        }
        self.scaler = torch.amp.GradScaler(enabled=autocast_enabled)

        self.beta_scene = beta_scene
        self.beta_motion = beta_motion
        self.best_val_loss = float('inf')

        self.save_path = save_path
        if os.path.exists(self.save_path)==False:
            os.makedirs(self.save_path)

    def train_one_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        with torch.autocast(**self.autocast_config):
            scene_token_ids = batch['scene_token'].to(self.device)
            motions = batch['motion'].to(self.device)
            out = self.model(scene_token_ids=scene_token_ids, motions=motions)
            loss = out['scene_loss'] * self.beta_scene + out['motion_loss'] * self.beta_motion

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "scene_loss": out["scene_loss"].detach(),
            "motion_loss": out["motion_loss"].detach(),
            "loss_total": loss.detach(),
        }

    def valid_one_step(self, batch):
        self.model.eval()
        with torch.autocast(**self.autocast_config):
            scene_token_ids = batch["scene_token"].to(self.device)
            motions = batch["motion"].to(self.device)
            out = self.model(scene_token_ids=scene_token_ids, motions=motions)
            loss = out["scene_loss"] * self.beta_scene + out["motion_loss"] * self.beta_motion

        return {
            "scene_loss": out["scene_loss"].detach(),
            "motion_loss": out["motion_loss"].detach(),
            "loss_total": loss.detach(),
        }

    def train(self, log_interval=50, val_interval=1):
        for epoch in range(self.num_epochs):
            train_loss_sum = 0.0
            train_scene_sum = 0.0
            train_motion_sum = 0.0

            for step, batch in enumerate(self.train_dl):
                log = self.train_one_step(batch)

                train_loss_sum += log["loss_total"].item()
                train_scene_sum += log["scene_loss"].item()
                train_motion_sum += log["motion_loss"].item()

                if (step + 1) % log_interval == 0:
                    n = step + 1
                    avg_total = train_loss_sum / n
                    avg_scene = train_scene_sum / n
                    avg_motion = train_motion_sum / n

                    print(
                        f"[Epoch {epoch+1} | Step {step+1}] "
                        f"train_loss={avg_total:.4f} | "
                        f"scene_loss={avg_scene:.4f} | "
                        f"motion_loss={avg_motion:.4f}"
                    )

            if (epoch + 1) % val_interval == 0:
                self.model.eval()

                val_loss_sum = 0.0
                val_scene_sum = 0.0
                val_motion_sum = 0.0

                with torch.no_grad():
                    for batch in self.valid_dl:
                        log = self.valid_one_step(batch)

                        val_loss_sum += log["loss_total"].item()
                        val_scene_sum += log["scene_loss"].item()
                        val_motion_sum += log["motion_loss"].item()

                n_val = max(1, len(self.valid_dl))
                val_avg = val_loss_sum / n_val
                val_scene_avg = val_scene_sum / n_val
                val_motion_avg = val_motion_sum / n_val

                print(
                    f"[Epoch {epoch+1}] "
                    f"val_loss={val_avg:.4f} | "
                    f"scene_loss={val_scene_avg:.4f} | "
                    f"motion_loss={val_motion_avg:.4f}"
                )

                if val_avg < self.best_val_loss:
                    print(
                        f"Validation loss improved from {self.best_val_loss:.4f} "
                        f"to {val_avg:.4f}. Saving best model..."
                    )
                    self.best_val_loss = val_avg

                    save_path = os.path.join(self.save_path, "best_model.pth")
                    torch.save(self.model.state_dict(), save_path)

                last_path = os.path.join(self.save_path, "last_model.pth")
                torch.save(self.model.state_dict(), last_path)