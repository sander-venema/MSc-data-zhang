import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image

from utils.losses import BCEDiceLoss, IoULoss, DiceLoss

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from sklearn.metrics import jaccard_score

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
    outputs = outputs.view(-1)
    targets = targets.view(-1)

    IoU = jaccard_score(targets, outputs)
    return IoU

# Define the path to the dataset
dataset_path = "new_dataset"

# Define the path to the images and masks
images_path = f"{dataset_path}/images"
masks_path = f"{dataset_path}/labels"

# Define the batch size
batch_size = 8

# Define the number of epochs
num_epochs = 100

# Define the device to use for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transforms to apply to the images and masks
image_transforms = transforms.Compose([
    transforms.Resize((369, 369)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transforms = transforms.Compose([
    transforms.Resize((369, 369)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.round(x)),
])

# Load the dataset
dataset = ImageFolder(images_path, transform=image_transforms)

# Load the masks
masks = ImageFolder(masks_path, transform=mask_transforms)

# Combine the images and masks into a single dataset
dataset = [(image, mask) for (image, _), (mask, _) in zip(dataset, masks)]

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
criterion = BCEDiceLoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

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
