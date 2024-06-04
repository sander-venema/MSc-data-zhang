import os
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms

from backbones_unet.model.unet import Unet
from backbones_unet.utils.dataset import SemanticSegmentationDataset

from utils.metrics import DiceCoefficient, PixelAccuracy, mIoU
from sklearn.metrics import precision_score, recall_score

from archs.model_wass import Generator_3, Discriminator_3
from utils.data_stuff import GenerationDataset
from tqdm import tqdm

def main(enable_gan):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if enable_gan:
        print("GAN enabled")
        G = Generator_3().to("cuda")
        D = Discriminator_3().to("cuda")

        pretrained_gan = "saved_models/generation/wassGPaug32_5e-05_5fold_1_best.pt"
        G.load_state_dict(torch.load(pretrained_gan)["G"], strict=False)
        D.load_state_dict(torch.load(pretrained_gan)["D"], strict=False)

        gan_dataset = GenerationDataset(root_dir="extra_data/test/images/", transform=transform)
        gan_loader = DataLoader(gan_dataset, batch_size=1, shuffle=False)
        it = iter(gan_loader)

    model = Unet(
        backbone='vgg16_bn',
        in_channels=3,
        num_classes=1,
    )

    pretrained_segmentation = "saved_models/segmentation/unet_vgg16_bn_dice_bce_0.0001_comb_best.pth"
    model.load_state_dict(torch.load(pretrained_segmentation))
    model.to("cuda")
    model.eval()

    seg_dataset = SemanticSegmentationDataset("extra_data/test/images", "extra_data/test/labels")
    seg_loader = DataLoader(seg_dataset, batch_size=1, shuffle=False)

    # Initialize running means for metrics
    running_iou = 0
    running_dice = 0
    running_precision = 0
    running_recall = 0
    count = 0

    fake_skip = 0

    for i, (images, masks) in enumerate(seg_loader):
        images = images.to("cuda")
        masks = masks.to("cuda")

        if enable_gan:
            # Get first image from gan_loader
            img = next(it).to("cuda")

            # Let discriminator decide if the image is real or fake, skip if it believes it's fake
            confidence = D(img)
            real = torch.round(torch.sigmoid(confidence))

            if real.item() != 1:
                fake_skip += 1
                continue

        # Segment the image
        seg_output = model(images)
        seg_output = (seg_output > 0.5).float()

        # Update running means
        running_iou += mIoU(seg_output, masks)
        running_dice += DiceCoefficient(seg_output, masks)
        running_precision += precision_score(masks.cpu().numpy().flatten(), seg_output.cpu().numpy().flatten(), zero_division=0)
        running_recall += recall_score(masks.cpu().numpy().flatten(), seg_output.cpu().numpy().flatten(), zero_division=0)
        count += 1

    if count == 0:
        print("No real images found")
        count = 1

    print(f"Mean IoU: {running_iou / count}\n")
    print(f"Mean Dice: {running_dice / count}\n")
    print(f"Mean Precision: {running_precision / count}\n")
    print(f"Mean Recall: {running_recall / count}\n")
    if enable_gan:
        print(f"Skipped {fake_skip} fake images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enable or disable GAN in the pipeline.")
    parser.add_argument('-G', '--enable_gan', action='store_true', help='Enable GAN if specified, otherwise disable.')
    args = parser.parse_args()

    main(args.enable_gan)
