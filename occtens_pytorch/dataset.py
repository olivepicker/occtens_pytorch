import os
import torch
import numpy as np
import json
import random
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from pyquaternion import Quaternion

class SceneDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        d = self.df.iloc[idx]
        path = d.path
        scene_id = d.scene
        curr_id = d.scene_id
        f = np.load(os.path.join('data', path))

        sem = f['semantics']
        mask_lidar = f['mask_lidar']
        mask_camera = f['mask_camera']
        
        sem, mask_lidar, mask_camera = map(lambda t: rearrange(t, 'x y z -> z y x'), (sem, mask_lidar, mask_camera))
        valid_mask = (mask_lidar > 0)

        if self.transform is not None:
            sem, valid_mask = self.transform(sem, valid_mask)
        # sem[~valid_mask] = 18

        out = {
            'semantic': torch.tensor(sem).float(),
            'mask': torch.tensor(valid_mask).long(),
            'scene_num': scene_id,
            'scene_id': curr_id
        }
        return out

class VoxelAugmentor:
    def __init__(
        self, 
        flip_prob: float = 0.5, 
        rot_prob: float = 0.5,
        rot_range: float = 45.0,
        scale_prob: float = 0.5,
        scale_range: tuple = (0.9, 1.1),
        empty_idx: int = 17,
        ignore_idx: int = 255
    ):
        self.flip_prob = flip_prob
        self.rot_prob = rot_prob
        self.rot_range = rot_range
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.empty_idx = empty_idx
        self.ignore_idx = ignore_idx

    def __call__(self, voxel, mask=None):
        if isinstance(voxel, np.ndarray):
            voxel = torch.from_numpy(voxel).long()
        if mask is not None and isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).bool()

        if random.random() < self.flip_prob:
            voxel = torch.flip(voxel, dims=[-1])
            if mask is not None:
                mask = torch.flip(mask, dims=[-1])

        do_rot = random.random() < self.rot_prob
        do_scale = random.random() < self.scale_prob

        if do_rot or do_scale:
            angle = random.uniform(-self.rot_range, self.rot_range) if do_rot else 0.0
            scale = random.uniform(*self.scale_range) if do_scale else 1.0
            
            voxel, mask = self._apply_affine(voxel, mask, angle, scale)

        return voxel, mask

    def _apply_affine(self, voxel, mask, angle_deg, scale):
        inp_voxel = voxel.unsqueeze(0).unsqueeze(0).float()
        
        theta = np.deg2rad(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        s = 1.0 / scale
        
        affine_matrix = torch.tensor([
            [s * cos_t, s * -sin_t, 0, 0],
            [s * sin_t, s * cos_t,  0, 0],
            [0,         0,          1, 0]
        ], dtype=torch.float32, device=voxel.device).unsqueeze(0)

        d, h, w = voxel.shape
        grid = F.affine_grid(affine_matrix, [1, 1, d, h, w], align_corners=False)

        aug_voxel = F.grid_sample(inp_voxel, grid, mode='nearest', padding_mode='zeros', align_corners=False)
        aug_voxel = aug_voxel.squeeze().long()

        if mask is not None:
            inp_mask = mask.unsqueeze(0).unsqueeze(0).float()
            aug_mask = F.grid_sample(inp_mask, grid, mode='nearest', padding_mode='zeros', align_corners=False)
            aug_mask = aug_mask.squeeze().bool()
            
            aug_voxel[~aug_mask] = self.empty_idx
            mask = aug_mask

        return aug_voxel, mask

def get_yaw_from_quaternion(q_list):
    q = Quaternion(q_list)
    yaw, pitch, roll = q.yaw_pitch_roll
    return yaw


def get_relative_motion(prev_pose, curr_pose):
    prev_trans = np.array(prev_pose['translation'][:2])
    prev_yaw = get_yaw_from_quaternion(prev_pose['rotation'])

    curr_trans = np.array(curr_pose['translation'][:2])
    curr_yaw = get_yaw_from_quaternion(curr_pose['rotation'])

    d_theta = curr_yaw - prev_yaw
    d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
    delta_global = curr_trans - prev_trans
    
    c, s = np.cos(prev_yaw), np.sin(prev_yaw)
    R_inv = np.array([[c, s], [-s, c]])
    
    d_xy_local = R_inv @ delta_global
    
    dx = d_xy_local[0]
    dy = d_xy_local[1]

    return dx, dy, d_theta


def load_annotation(ann_path):
    with open(ann_path) as f:
        ann = json.load(f)

    return ann

def get_yaw_from_quaternion(q_list):
    q = Quaternion(q_list)
    yaw, pitch, roll = q.yaw_pitch_roll
    return yaw


def get_relative_motion(prev_pose, curr_pose):
    prev_trans = np.array(prev_pose['translation'][:2])
    prev_yaw = get_yaw_from_quaternion(prev_pose['rotation'])

    curr_trans = np.array(curr_pose['translation'][:2])
    curr_yaw = get_yaw_from_quaternion(curr_pose['rotation'])

    d_theta = curr_yaw - prev_yaw
    d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
    delta_global = curr_trans - prev_trans
    
    c, s = np.cos(prev_yaw), np.sin(prev_yaw)
    R_inv = np.array([[c, s], [-s, c]])
    
    d_xy_local = R_inv @ delta_global
    
    dx = d_xy_local[0]
    dy = d_xy_local[1]

    return dx, dy, d_theta


class OccTENSDataset(Dataset):
    def __init__(self, df, ann_path, num_frames=10, token_map_path='scene_output/tokens'):
        self.df = df
        self.scene_id = self.df.scene.unique()
        self.ann = load_annotation(ann_path)
        self.num_frames = num_frames
        self.token_map_path = token_map_path

    def __len__(self):
        return len(self.scene_id)
    
    def __getitem__(self, idx):
        scene_id = self.scene_id[idx]

        d = self.df[self.df['scene']==scene_id].reset_index(drop=True).sort_values('timestamp', ascending=True)
        tokens = []
        motions = []

        max_index = len(d) - self.num_frames
        start = np.random.randint(0, max_index+1)
        
        for i in range(start, start + self.num_frames):
            scene_id = d.iloc[i].scene
            curr_id = d.iloc[i].scene_id
            token = np.load(os.path.join(self.token_map_path, f'{scene_id}_{curr_id}.npy'))
            
            ann = self.ann['scene_infos'][scene_id]
            curr_ann = ann[curr_id]
            curr_pose = curr_ann['ego_pose']

            prev_id = curr_ann['prev']
            prev_pose = ann[prev_id]['ego_pose'] if prev_id != 'EOF' else None

            if prev_pose is not None:
                x, y, theta = get_relative_motion(prev_pose, curr_pose)

            else:
                x, y, theta = 0, 0, 0    

            tokens.append(token); motions.append([x, y, theta])

        out = {
            #'semantic': torch.tensor(sem).float(),
            #'mask': torch.tensor(valid_mask).long(),
            'scene_token': torch.tensor(np.array(tokens)).long(),
            'motion': torch.tensor(np.array(motions)).float(),
            'scene_num': scene_id,
            'scene_id': curr_id
        }

        return out