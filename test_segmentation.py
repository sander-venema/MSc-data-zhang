import os
import torch
import numpy as np
from backbones_unet.model.unet import Unet
from backbones_unet.utils.dataset import SemanticSegmentationDataset

from skimage import color

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

for i, (images, masks) in enumerate(test_loader):
    images = images.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        outputs = model(images)
        outputs = torch.sigmoid(outputs)
        outputs = (outputs > 0.5).float()

    image = images.cpu().numpy()
    mask = outputs.cpu().numpy()

    plt.imshow(image[0].transpose(1, 2, 0), cmap='gray')
    plt.imshow(mask[0].transpose(1, 2, 0), cmap='autumn', alpha=0.5)
    plt.imshow(masks[0].cpu().numpy().transpose(1, 2, 0), cmap='winter', alpha=0.5)
    plt.show()
