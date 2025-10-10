from torch import einsum
from utils import simplex, sset
import torch
import torch.nn as nn
from losses import CrossEntropy

class DiceLoss(nn.Module):
    def __init__(self, **kwargs):
        super(DiceLoss, self).__init__()
        self.idk = kwargs.get('idk', None)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def forward(self, pred_softmax, target):
        assert simplex(pred_softmax)
        assert sset(target, [0, 1])

        if self.idk is not None:
            pred_softmax = pred_softmax[:, self.idk, ...]
            target = target[:, self.idk, ...]

        # Dice coefficient calculation
        intersection = torch.sum(pred_softmax * target, dim=(2, 3))
        cardinality = torch.sum(pred_softmax + target, dim=(2, 3))
        
        dice_score = (2. * intersection) / (cardinality + 1e-6)
        
        return 1. - torch.mean(dice_score)

class CombinedLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CombinedLoss, self).__init__()
        self.idk = kwargs.get('idk', None)
        self.cross_entropy_loss = CrossEntropy(**kwargs)
        self.dice_loss = DiceLoss(**kwargs)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def forward(self, pred_softmax, target):
        ce_loss = self.cross_entropy_loss(pred_softmax, target)
        dice_loss = self.dice_loss(pred_softmax, target)
        return ce_loss + dice_loss