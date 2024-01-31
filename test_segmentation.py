import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks
from backbones_unet.model.unet import Unet
from backbones_unet.utils.dataset import SemanticSegmentationDataset

import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

import argparse

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

parser = argparse.ArgumentParser(description='Store test settings')
parser.add_argument('--model', type=str, default='unet_vgg16_dice_bce_0.0001.pth', help='Model name')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = 'new_dataset/test'
output_path = 'output_directory'

if not os.path.exists(output_path):
    os.makedirs(output_path)

dataset = SemanticSegmentationDataset(os.path.join(dataset_path, 'images'), os.path.join(dataset_path, 'labels'))
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

model = Unet(
    backbone='vgg16',
    in_channels=3,
    num_classes=1,
)

model.load_state_dict(torch.load("saved_models/segmentation/" + args.model))
model.to(device)
model.eval()

resize = transforms.Resize(size=(256, 256))
# Iterate through each image in the dataset

for i, (images, masks) in enumerate(test_loader):
    images = images.to(device)
    images = images.unsqueeze(0)
    output = model(images)
    output = torch.sigmoid(output)
    images = (images*255).to(torch.uint8)
    output = output.cpu()
    stack_mask = draw_segmentation_masks(images, masks=output, alpha=0.5)
    show(stack_mask)
