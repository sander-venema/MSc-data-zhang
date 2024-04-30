import os
import torch
import numpy as np
from backbones_unet.model.unet import Unet
from backbones_unet.utils.dataset import SemanticSegmentationDataset

from utils.metrics import DiceCoefficient, PixelAccuracy, mIoU
from sklearn.metrics import precision_score, recall_score

from skimage import color

import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from torch.utils.data import DataLoader

models_dir = "saved_models/segmentation/"
models = [f for f in os.listdir(models_dir) if f.endswith("comb_best.pth")]

count = 0
for model_name in models:
    print(f"Testing model {count + 1}/{len(models)} with name {model_name}\n")
    count += 1
    if "resnet101" in model_name:
        print("Using resnet101\n")
        model = deeplabv3_resnet101(weights="DeepLabV3_ResNet101_Weights.DEFAULT")
        model.classifier = DeepLabHead(2048, 1)
    elif "unet_resnext101" in model_name:
        print("Using resnext101_32x8d\n")
        model = Unet(
            backbone='resnext101_32x8d',
            in_channels=3,
            num_classes=1,
        )
    elif "unet_vgg16_bn" in model_name:
        print("Using vgg16_bn\n")
        model = Unet(
            backbone='vgg16_bn',
            in_channels=3,
            num_classes=1,
        )
    elif "unet_vgg16" in model_name:
        print("Using vgg16\n")
        model = Unet(
            backbone='vgg16',
            in_channels=3,
            num_classes=1,
        )
    elif "unet_vgg19_bn" in model_name:
        print("Using vgg19_bn\n")
        model = Unet(
            backbone='vgg19_bn',
            in_channels=3,
            num_classes=1,
        )
    else:
        print("Using vgg19\n")
        model = Unet(
            backbone='vgg19',
            in_channels=3,
            num_classes=1,
        )

    model.load_state_dict(torch.load(os.path.join(models_dir, model_name)))
    model.to("cuda")
    model.eval()

    dataset = SemanticSegmentationDataset('combined_dataset/test/images', 'combined_dataset/test/labels')
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    running_dice = 0.0
    running_pixel_accuracy = 0.0
    running_miou = 0.0
    running_precision = 0.0
    running_recall = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images = images.to("cuda")
            masks = masks.to("cuda")
            if "resnet101" in model_name:
                outputs = model(images)["out"]
            else:
                outputs = model(images)

            # outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float()

            running_dice += DiceCoefficient(outputs, masks)
            running_pixel_accuracy += PixelAccuracy(outputs, masks)
            running_miou += mIoU(outputs, masks)
            running_precision += precision_score(masks.cpu().numpy().flatten(), outputs.cpu().numpy().flatten(), zero_division=0)
            running_recall += recall_score(masks.cpu().numpy().flatten(), outputs.cpu().numpy().flatten(), zero_division=0)

            num_batches += 1

    running_dice /= num_batches
    running_pixel_accuracy /= num_batches
    running_miou /= num_batches
    running_precision /= num_batches
    running_recall /= num_batches

    # Write running averages to file
    with open("accuracy_seg.txt", "a+") as f:
        f.write(f"{model_name}: ")
        f.write(f"Dice Coefficient: {running_dice}, ")
        f.write(f"Pixel Accuracy: {running_pixel_accuracy}, ")
        f.write(f"mIoU: {running_miou}, ")
        f.write(f"Precision: {running_precision}, ")
        f.write(f"Recall: {running_recall}\n")