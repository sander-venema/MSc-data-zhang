import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, outputs, targets):
        smooth = 1e-5

        intersection = (outputs * targets).sum()
        union = outputs.sum() + targets.sum()

        dice_coefficient = (2. * intersection + smooth) / (union + smooth)
        return 1. - dice_coefficient

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    return iou

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
# model.classifier = DeepLabHead(2048, 1)

# Define the loss function
criterion = BCEDiceLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Define the scheduler
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

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
    total_white_pixels = 0
    correct_white_pixels = 0
    iou_running = 0.0

    # Iterate over the validation data
    for i, (images, masks) in enumerate(val_loader):
        # Move the images and masks to the device
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)["out"]
        iou_running += iou_score(outputs, masks)

        for j in range(len(outputs)):
            output = outputs[j]
            output_binary = (output > 0.5).float()
            correct_white_pixels += (output_binary == masks).logical_and(masks == 1).sum().item()
            total_white_pixels += masks.eq(1).sum().item()
       
        if i%10 == 0:
            print(f"Validation Batch {i + 1}/{len(val_loader)} Accuracy: {correct_white_pixels / total_white_pixels:.4f}")

        # # Save the predicted mask to png files
        # for j in range(len(outputs)):
        #     output = outputs[j]
        #     output = (output > 0.5).float()
        #     output = output.to("cpu").numpy()
        #     output = np.uint8(output * 255)
        #     output = Image.fromarray(output[0], mode="L")
        #     output.save(f"output_segmentation/output_{i * batch_size + j}.png") 

        # Calculate the pixel accuracy
        # correct_pixels += ((outputs > 0.5) == masks).sum().item()
        # total_pixels += masks.numel()

    # Compute the pixel accuracy
    # pixel_accuracy = correct_pixels / total_pixels
    pixel_accuracy = correct_white_pixels / total_white_pixels

    # Add the pixel accuracy to the TensorBoard writer
    writer.add_scalar("Pixel_Accuracy/val", pixel_accuracy, epoch)

    # Get IoU score
    iou_val = iou_running / len(val_loader)
    writer.add_scalar("IoU/val", iou_val, epoch)

    # Print the epoch number, loss, and accuracy
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Pixel Accuracy: {pixel_accuracy:.4f}")
