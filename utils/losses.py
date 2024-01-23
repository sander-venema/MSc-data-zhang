import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.lovasz_losses import lovasz_hinge, binary_xloss

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

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, outputs, targets):
        loss = lovasz_hinge(outputs, targets, per_image=False)
        return loss
    
class Binary_Xloss(nn.Module):
    def __init__(self):
        super(Binary_Xloss, self).__init__()

    def forward(self, outputs, targets):
        loss = binary_xloss(outputs, targets)
        return loss

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
    