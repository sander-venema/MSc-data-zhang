import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image

from utils.losses import BCEDiceLoss, IoULoss, DiceLoss, LovaszHingeLoss, Binary_Xloss, FocalLoss
from utils.metrics import DiceCoefficient, PixelAccuracy, mIoU
from utils.data_stuff import SegmentationDataset, image_transforms, mask_transforms

from torchvision.models.segmentation import deeplabv3_resnet101

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Store training settings')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--loss', type=int, default=0, help='Loss function; 0: BCEDiceLoss, \
                    1: IoULoss, 2: DiceLoss, 3: LovaszHingeLoss, 4: Binary_Xloss, 5: FocalLoss, 6: BCELoss')

LOSSES = [BCEDiceLoss(), IoULoss(), DiceLoss(), LovaszHingeLoss(), Binary_Xloss(), FocalLoss(), F.binary_cross_entropy_with_logits()]

args = parser.parse_args()

BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SegmentationDataset('new_dataset/', image_transforms, mask_transforms)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = deeplabv3_resnet101(weights="DeepLabV3_ResNet101_Weights.DEFAULT")

model.classifier = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

criterion = LOSSES[args.loss]

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

model.to(device)

loss_short = 'bce_dice' if args.loss == 0 else 'iou' if args.loss == 1 else 'dice' if args.loss == 2 else 'lovasz' if args.loss == 3 else 'bce_xloss' if args.loss == 4 else 'focal' if args.loss == 5 else 'bce'
run_name = "resnet101_{0}_{1}".format(loss_short, LEARNING_RATE)

writer = SummaryWriter(f"logs_segmentation/{run_name}")

num_epochs = 100
best_val_loss = float("inf")

print(f"Training {run_name} for {num_epochs} epochs with batch size {BATCH_SIZE}, learning rate {LEARNING_RATE} and loss {loss_short}")

# Train the model
for epoch in tqdm(range(num_epochs)):
    # Set the model to training mode
    model.train()

    # Initialize the running loss
    running_loss = 0.0

    # Iterate over the training data
    for i, (images, masks) in enumerate(train_loader):
        # Move the images and masks to the device
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)["out"]
        loss = criterion(outputs, masks.to(torch.float32))

        if i%10 == 0:
            print(f"Training Batch {i + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Compute the average loss
    avg_loss = running_loss / len(train_loader)

    # Add the loss to the TensorBoard writer
    writer.add_scalar("Loss/train", avg_loss, epoch)

    # Set the model to evaluation mode
    model.eval()

    # Initialize the running accuracy
    dice_running = 0.0
    pixel_accuracy_running = 0.0
    iou_running = 0.0
    val_loss_running = 0.0
    length = 0

    # Iterate over the validation data
    for i, (images, masks) in enumerate(val_loader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)["out"]
        val_loss_running += criterion(outputs, masks.to(torch.float32)).item()

        for j in range(len(outputs)):
            length += 1
            output = outputs[j]
            output = (output > 0.5).float()
            dice_running += DiceCoefficient(output, masks[j])
            pixel_accuracy_running += PixelAccuracy(output, masks[j])
            iou_running += mIoU(output, masks[j])

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
            
    # Compute average Dice Coefficient
    dice_val = dice_running / length
    writer.add_scalar("Metrics/Dice", dice_val, epoch)

    # Compute average IoU
    iou_val = iou_running / length
    writer.add_scalar("Metrics/IoU", iou_val, epoch)

    val_loss = val_loss_running / len(val_loader)
    scheduler.step()
    writer.add_scalar("Loss/val", val_loss, epoch)

    # Save the model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"saved_models/segmentation/{run_name}.pth")
