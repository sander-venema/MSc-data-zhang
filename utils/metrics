from sklearn.metrics import jaccard_score

def DiceCoefficient(outputs, targets):
    outputs = outputs.view(-1)
    targets = targets.view(-1)

    intersection = (outputs * targets).sum()
    dice = (2. * intersection + 1e-5) / (outputs.sum() + targets.sum() + 1e-5)
    return dice

def PixelAccuracy(outputs, targets):
    outputs = outputs.view(-1)
    targets = targets.view(-1)

    correct = (outputs == targets).sum()
    total = outputs.size(0)
    return correct / total

def mIoU(outputs, targets):
    outputs = outputs.view(-1).to("cpu").numpy()
    targets = targets.view(-1).to("cpu").numpy()

    IoU = jaccard_score(targets, outputs)
    return IoU