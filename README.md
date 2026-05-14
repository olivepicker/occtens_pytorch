## OccTENS (wip)

Unofficial implementation proposed [OccTENS: 3D Occupancy World Model via Temporal Next-Scale Prediction](https://arxiv.org/abs/2509.03887) from Jin et al.

## TODO
- [x] **Scene Tokenizer**
    - [x] Implement VQ-VAE
        - [x] *Residual Block*
    - [x] Multi-Scale Quantizer
        - [x] *Develop Phi*
        - [x] *Normalize* 
        - [x] *Attention Layer in Encoder / Decoder*
- [x] **Motion Tokenizer**
- [x] **World Model**
    - [x] Implement TENSFormer
        - [x] *Attention Mask - Temporal, Spatial*
    - [x] Temporal Scene-by-scene Prediction
    - [x] Spatial Scale-by-scale Generation
    - [x] Multi-modal Camera Pose Aggregation
    - [x] Auto-Regressive Wrapper
- [x] **Train / Inference Pipeline**
    - [x] Implement Losses
    - [x] Train base model
    - [x] Generate
    - [x] Scale-wise generation
    - [ ] Validate forecasting performance

## Usage
```python
# 1. Train Scene Tokenizer

from networks.scene_tokenizer import MultiScaleVQVAE
from trainer import SceneTokenizerTrainer
from dataset import SceneDataset
...

NUM_FRAMES = 10
CONTEXT_FRAME_POINT = 4  

m = MultiScaleVQVAE(
    in_channels = 288
)

df = pd.read_csv('data/scene_annotations.csv')

train_df = df[df['is_train']==True].reset_index(drop=True)
valid_df = df[df['is_train']==False].reset_index(drop=True)

train_ds = SceneDataset(df=train_df)
valid_ds = SceneDataset(df=valid_df)

trainer = SceneTokenizerTrainer(
    num_epochs = 50,
    model = m,
    optimizer = torch.optim.AdamW(lr=5e-5,params=m.parameters()),
    train_ds = train_ds,
    valid_ds = valid_ds,
    batch_size = 8,
    device = 'cuda',
    autocast_enabled = True,    # Now autocast works!
    ignore_index = 255,
)


trainer.train()
```

```python
# 2. Save token maps
...

trainer = SceneTokenizerTrainer(
    num_epochs = None,
    model = m,          # Load the best weights before run!
    optimizer = None,
    train_ds = train_ds, 
    valid_ds = valid_ds,
    batch_size = 8,
    device = 'cuda',
    autocast_enabled = True,
    save_token = True   # If True, valid_ds will be concatenated with train_ds to tokenize the entire dataset.
)

trainer.save_token_all()
```

```python
# 3. Train OccTENS (Using Occ3D Nuscenes Dataset)

from occtens_pytorch import OccTENS
from trainer import OccTENSTrainer
from dataset import OccTENSDataset

...
train_ds = OccTENSDataset(df = train_df, ann_path = 'data/annotations.json', num_frames = NUM_FRAMES) 
valid_ds = OccTENSDataset(df = valid_df, ann_path = 'data/annotations.json', num_frames = NUM_FRAMES)

m = OccTENS(
    dim = 128
)

trainer = OccTENSTrainer(       # AutoRegressive Training
    num_epochs = 50,
    model = m,
    optimizer = torch.optim.AdamW(lr=1e-4, params=m.parameters()),
    train_ds = train_ds,
    valid_ds = valid_ds,
    batch_size = 2,
    device = 'cuda',
    autocast_enabled = True,    # Now autocast works!
    context_frame_point = CONTEXT_FRAME_POINT
)

trainer.train()
```

```python
# 4. Generate Tokens

m = OccTENS(
    dim = 128
)

print(m.load_state_dict(torch.load('weights/occtens_best_model.pth')))

m.eval()
ar = AutoRegressiveWrapper(
    model=m
)

o = valid_ds[0] # test example (pick 1 sample)
scene_token_ids = o['scene_token'][None,...]
motions = o['motion'][None,...]

ar_out = ar.generate(scene_token_ids.to('cuda:0'), motions.to('cuda:0'))


# 5. Reconstruction

def tokens_to_indices_list(token_seq, scales):
    B, T_total = token_seq.shape
    indices_list = []
    offset = 0

    for s in scales:
        n = s * s
        slice_ = token_seq[:, offset:offset + n]      # (B, n)
        idx_s = slice_.view(B, s, s)                  # (B, s, s)
        indices_list.append(idx_s)
        offset += n

    assert offset == T_total, "Token length != sum_s s*s"
    return indices_list

vq = MultiScaleVQVAE(
    in_channels = 288
).to('cuda:0')

vq.load_state_dict(torch.load('weights/scene_best_model.pth'))

logits = []
for i in range(NUM_FRAMES):
    indices_list = tokens_to_indices_list(token_seq=ar_out['scene_token_ids'][:,i,:].to('cuda:0'), scales=(1,5,10,15,20,25))
    out = vq.decode_from_indices(indices_list, out_zyx=True)
    logits.append(out.detach().cpu().numpy())
```

## Citations

```bibtex
@misc{jin2025occtens3doccupancyworld,
      title={OccTENS: 3D Occupancy World Model via Temporal Next-Scale Prediction}, 
      author={Bu Jin and Songen Gu and Xiaotao Hu and Yupeng Zheng and Xiaoyang Guo and Qian Zhang and Xiaoxiao Long and Wei Yin},
      year={2025},
      eprint={2509.03887},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.03887}, 
}
```
```bibtex
@misc{tian2024visualautoregressivemodelingscalable,
      title={Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction}, 
      author={Keyu Tian and Yi Jiang and Zehuan Yuan and Bingyue Peng and Liwei Wang},
      year={2024},
      eprint={2404.02905},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.02905}, 
}
```
