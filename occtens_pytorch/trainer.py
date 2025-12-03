import torch
import torch.nn as nn
import torch.nn.functional as F

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

class OccTENSTrainer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.model = AutoRegressiveWrapper(model)

    def train_one_step(self):
        self.model.train()
        pass

    def valid_one_step(self):
        self.model.eval()
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