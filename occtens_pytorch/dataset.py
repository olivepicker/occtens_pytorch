import os
import torch
import numpy as np
import json

from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from pyquaternion import Quaternion

class SceneDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        d = self.df.iloc[idx]
        path = d.path
        f = np.load(os.path.join('data', path))
        sem = f['semantics']
        mask_lidar = f['mask_lidar']
        mask_camera = f['mask_camera']
        
        sem, mask_lidar, mask_camera = map(lambda t: rearrange(t, 'x y z -> z y x'), (sem, mask_lidar, mask_camera))
        valid_mask = (mask_lidar > 0)
        # sem[~valid_mask] = 18

        out = {
            'semantic': torch.tensor(sem).float(),
            'mask': torch.tensor(valid_mask).long(),
        }

        return out
    

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


def load_annotation(ann_path):
    with open(ann_path) as f:
        ann = json.load(f)

    return ann


class SceneTokenPrepareDataset(Dataset):
    def __init__(self, df, ann_path):
        self.df = df
        self.ann = load_annotation(ann_path)

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
        valid_mask = (mask_lidar > 0) | (mask_camera > 0)
        
        ann = self.ann['scene_infos'][scene_id]
        curr_ann = ann[curr_id]
        curr_pose = curr_ann['ego_pose']

        prev_id = curr_ann['prev']
        prev_pose = ann[prev_id]['ego_pose'] if prev_id != 'EOF' else None

        if prev_pose is not None:
            x, y, theta = get_relative_motion(prev_pose, curr_pose)

        else:
            x, y, theta = 0, 0, 0    

        out = {
            'semantic': torch.tensor(sem).float(),
            'mask': torch.tensor(valid_mask).long(),
            'x': torch.tensor(x).float(),
            'y': torch.tensor(y).float(),
            'theta': torch.tensor(theta).float(),
            'scene_num': scene_id,
            'scene_id': curr_id
        }

        return out