import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models.segmentation import deeplabv3_resnet101
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Define the path to the dataset
dataset_path = "new_dataset"

# Define the path to the images and masks
images_path = f"{dataset_path}/images"
masks_path = f"{dataset_path}/labels"

# Define the batch size
batch_size = 4

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
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
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
model = deeplabv3_resnet101(pretrained=True)

# Replace the classifier with a new one
model.classifier = nn.Sequential(
    nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.Conv2d(256, 3, kernel_size=1, stride=1)

)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

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
        print(f"Batch {i + 1}/{len(train_loader)}")
        # Move the images and masks to the device
        images = images.to(device)
        masks = masks.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)["out"]
        # print(outputs.shape)
        # print(masks.shape)

        # Compute the loss
        loss = criterion(outputs, masks.to(torch.float32))

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
    total_pixels = 0
    correct_pixels = 0

    # Iterate over the validation data
    for i, (images, masks) in enumerate(val_loader):
        # Move the images and masks to the device
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)["out"]

        # Calculate the pixel accuracy
        # predicted_labels = outputs.argmax(1)
        # print(predicted_labels.shape)
        # print(masks.shape)
        correct_pixels += (outputs == masks).sum().item()
        total_pixels += masks.numel()

    # Compute the pixel accuracy
    pixel_accuracy = correct_pixels / total_pixels

    # Add the pixel accuracy to the TensorBoard writer
    writer.add_scalar("Pixel_Accuracy/val", pixel_accuracy, epoch)

    # Print the epoch number, loss, and accuracy
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Pixel Accuracy: {pixel_accuracy:.4f}")
