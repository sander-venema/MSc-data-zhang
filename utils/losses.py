import torch
import torch.nn as nn
from torch.nn import functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, outputs, targets):

        outputs = outputs.view(-1)
        targets = targets.view(-1)

        intersection = (outputs * targets).sum()
        dice = (2. * intersection + 1e-5) / (outputs.sum() + targets.sum() + 1e-5)
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

class BCEDiceFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        focal = (target * torch.log(input + smooth) + (1 - target) * torch.log(1 - input + smooth)) * (1 - input) ** 2
        focal = focal.sum() / num
        return 0.5 * bce + 0.5 * dice + focal

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, outputs, targets):
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        intersection = (outputs * targets).sum()
        total = (outputs + targets).sum()
        union = total - intersection

        IoU = (intersection + 1e-5) / (union + 1e-5)
        return 1 - IoU
    