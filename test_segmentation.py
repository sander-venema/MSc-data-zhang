import os

import torch
import numpy as np
from PIL import Image
import argparse

from backbones_unet.model.unet import Unet
from backbones_unet.utils.dataset import SemanticSegmentationDataset

parser = argparse.ArgumentParser(description='Store test settings')
parser.add_argument('--model', type=str, default='unet_vgg16_dice_bce_0.0001.pth', help='Model name')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SemanticSegmentationDataset('new_dataset/images', 'new_dataset/labels')

model = Unet(
    backbone='vgg16',
    in_channels=3,
    num_classes=1,
)

model.load_state_dict(torch.load("saved_models/segmentation/" + args.model))
model.to(device)
model.eval()

for i in range(len(dataset)):
    img, mask = dataset[i]
    img = img.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_mask = model(img)

    # Stack masks on images
    img = img.cpu().numpy()
    mask = mask.cpu().numpy()
    pred_mask = pred_mask.cpu().numpy()

    img = img.squeeze()
    mask = mask.squeeze()
    pred_mask = pred_mask.squeeze()

    img = np.moveaxis(img, 0, -1)
    mask = np.moveaxis(mask, 0, -1)
    pred_mask = np.moveaxis(pred_mask, 0, -1)

    stacked_array = np.concatenate((img, mask, pred_mask), axis=2)

    stacked_array = stacked_array * 255
    stacked_array = stacked_array.astype(np.uint8)

    output = Image.fromarray(stacked_array)
    output.save('output_segmentation/test_pred/' + str(i) + '.png')
