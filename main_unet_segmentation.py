import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import numpy as np
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

from utils.losses import BCEDiceLoss, IoULoss, DiceLoss, LovaszHingeLoss, Binary_Xloss, FocalLoss, BCELoss, DiceBCELoss
from utils.metrics import DiceCoefficient, PixelAccuracy, mIoU

from sklearn.metrics import precision_score, recall_score

from backbones_unet.model.unet import Unet
from backbones_unet.utils.dataset import SemanticSegmentationDataset

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Store training settings')
parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('-l', '--loss', type=int, default=7, help='Loss function; 0: BCEDiceLoss, 1: IoULoss, 2: DiceLoss, 3: LovaszHingeLoss, 4: Binary_Xloss, 5: FocalLoss, 6: BCELoss, 7: DiceBCELoss')
parser.add_argument('-m', '--model', type=str, default='vgg16_bn', help='Model backbone')

LOSSES = [BCEDiceLoss(), IoULoss(), DiceLoss(), LovaszHingeLoss(), Binary_Xloss(), FocalLoss(), BCELoss(), DiceBCELoss()]

args = parser.parse_args()

BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
BACKBONE = args.model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SemanticSegmentationDataset('new_dataset/train/images', 'new_dataset/train/labels', normalize=transforms.Normalize((0.5,), (0.5,)))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = Unet(
    backbone=BACKBONE,
    in_channels=3,
    num_classes=1
)

criterion = LOSSES[args.loss]

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(params, lr=LEARNING_RATE)

model.to(device)

loss_short = 'bce_dice' if args.loss == 0 else 'iou' if args.loss == 1 else 'dice' if args.loss == 2 else 'lovasz' if args.loss == 3 else 'bce_xloss' if args.loss == 4 else 'focal' if args.loss == 5 else 'bce' if args.loss == 6 else 'dice_bce'
run_name = "unet_{0}_{1}_{2}_norm".format(BACKBONE, loss_short, LEARNING_RATE)

writer = SummaryWriter(f"logs_segmentation/{run_name}")

num_epochs = 100
best_val_loss = float("inf")

print(f"Training {run_name} for {num_epochs} epochs with batch size {BATCH_SIZE}, learning rate {LEARNING_RATE} and loss {loss_short}")

cur_best_dice = 0.0
cur_best_iou = 0.0

for epoch in tqdm(range(num_epochs)):
    model.train()

    running_loss = 0.0

    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.to(torch.float32))

        if i%10 == 0:
            print(f"Training Batch {i + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    writer.add_scalar("Loss/train", avg_loss, epoch)

    model.eval()

    dice_running = 0.0
    pixel_accuracy_running = 0.0
    iou_running = 0.0
    val_loss_running = 0.0
    precision_running = 0.0
    recall_running = 0.0
    length = 0

    for i, (images, masks) in enumerate(val_loader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        val_loss_running += criterion(outputs, masks.to(torch.float32)).item()

        for j in range(len(outputs)):
            length += 1
            output = outputs[j]
            output = (output > 0.5).float()
            dice_running += DiceCoefficient(output, masks[j])
            pixel_accuracy_running += PixelAccuracy(output, masks[j])
            iou_running += mIoU(output, masks[j])
            precision_running += precision_score(masks[j].to("cpu").numpy().flatten(), output.to("cpu").numpy().flatten(), zero_division=0)
            recall_running += recall_score(masks[j].to("cpu").numpy().flatten(), output.to("cpu").numpy().flatten(), zero_division=0)

        if i%10 == 0:
            print(f"Validation Batch {i + 1}/{len(val_loader)}")

        # Save the predicted mask to png files
        for j in range(len(outputs)):
            output = outputs[j]
            output = (output > 0.5).float()
            output = output.to("cpu").numpy()
            output = np.uint8(output * 255)
            output = Image.fromarray(output[0], mode="L")
            output.save(f"output_segmentation/output_{i * BATCH_SIZE + j}.png") 

    pixel_accuracy = pixel_accuracy_running / length
    writer.add_scalar("Metrics/Pixel_acc", pixel_accuracy, epoch)
            
    dice_val = dice_running / length
    writer.add_scalar("Metrics/Dice", dice_val, epoch)

    iou_val = iou_running / length
    writer.add_scalar("Metrics/IoU", iou_val, epoch)

    precision_val = precision_running / length
    writer.add_scalar("Metrics/Precision", precision_val, epoch)

    recall_val = recall_running / length
    writer.add_scalar("Metrics/Recall", recall_val, epoch)

    val_loss = val_loss_running / len(val_loader)
    writer.add_scalar("Loss/val", val_loss, epoch)

    if dice_val > cur_best_dice and iou_val > cur_best_iou:
        with open(f"saved_models/segmentation/{run_name}_best.txt", "a+") as f:
            f.write(f"Epoch: {epoch}, Dice: {dice_val}, IoU: {iou_val}\n")
        cur_best_dice = dice_val
        cur_best_iou = iou_val
        torch.save(model.state_dict(), f"saved_models/segmentation/{run_name}_best.pth")
