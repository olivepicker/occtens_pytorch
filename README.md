## OccTENS (wip)

Unofficial implementation proposed [OccTENS: 3D Occupancy World Model via Temporal Next-Scale Prediction](https://arxiv.org/abs/2509.03887) from Jin et al.

## TODO
- [x] **Scene Tokenizer**
    - [x] Implement VQ-VAE
        - [x] *Residual Block*
    - [x] Multi-Scale Quantizer
        - [ ] *Develop Phi*
        - [ ] *Normalize* 
- [x] **Motion Tokenizer**
- [ ] **World Model**
    - [x] Implement TENSFormer
        - [x] *Attention Mask - Temporal, Spatial*
    - [ ] Temporal Scene-by-scene Prediction
    - [ ] Spatial Scale-by-scale Generation
    - [ ] Multi-modal Camera Pose Aggregation
    - [ ] Auto-Regressive Wrapper
- [ ] **Train / Inference Pipeline**
    - [x] Implement Losses
    - [ ] Train base model

## Usage
```python
# Train Scene Tokenizer

from networks.scene_tokenizer import MultiScaleVQVAE
from trainer import SceneTokenizerTrainer
...

m = MultiScaleVQVAE(
    in_channels=16
)

trainer = SceneTokenizerTrainer(
    model=m,
    optimizer = torch.optim.AdamW(lr=1e-4,params=m.parameters()),
    train_ds = train_ds,
    valid_ds = valid_ds,
    batch_size = 4,
    device = 'cuda',
)

trainer.train(num_epochs=200)
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
