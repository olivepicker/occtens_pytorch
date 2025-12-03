import torch
import torch.nn as nn
import torch.nn.functional as F

#wip
class SceneTokenizerTrainer():
    pass

class OccTENSTrainer():
    pass

class AutoRegressiveWrapper(nn.Module):
    def __init__(
        self,
        model
    ):
        super().__init__()
        self.model = model
    def forward(self,):
        #self.model()
        pass

    def generate(self, batch):
        pass