import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

from utils.losses import BCEDiceLoss, IoULoss, DiceLoss, FocalLoss
from utils.metrics import DiceCoefficient, PixelAccuracy, mIoU
from utils.data_stuff import SegmentationDataset, image_transforms, mask_transforms

from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Define the batch size
batch_size = 8

# Define the number of epochs
num_epochs = 100

# Define the device to use for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
dataset = SegmentationDataset('new_dataset/', image_transforms, mask_transforms)

# Split the dataset into training and validation sets
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

# Define the data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load the pre-trained Deeplabv3 ResNet 101 model
model = deeplabv3_resnet101(weights="DeepLabV3_ResNet101_Weights.DEFAULT")

# Replace the classifier with a new one
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

# Define the loss function
criterion = FocalLoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, verbose=True)

# Move the model to the device
model.to(device)

# Define the TensorBoard writer
writer = SummaryWriter()

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

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)["out"]

        # Compute the loss
        loss = criterion(outputs, masks.to(torch.float32))
        if i%10 == 0:
            print(f"Training Batch {i + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Update the running loss
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
    length = 0

    # Iterate over the validation data
    for i, (images, masks) in enumerate(val_loader):
        # Move the images and masks to the device
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)["out"]

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
            output.save(f"output_segmentation/output_{i * batch_size + j}.png") 

    pixel_accuracy = pixel_accuracy_running / length
    print(pixel_accuracy)
    writer.add_scalar("Metrics/Pixel_acc", pixel_accuracy, epoch)
            
    # Compute average Dice Coefficient
    dice_val = dice_running / length
    writer.add_scalar("Metrics/Dice", dice_val, epoch)

    # Compute average IoU
    iou_val = iou_running / length
    writer.add_scalar("Metrics/IoU", iou_val, epoch)

    # Print the epoch number, loss, and accuracy
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Dice: {dice_val:.4f}")
