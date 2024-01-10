import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from model_new import Generator, Discriminator

# Define constants
IMAGE_SIZE = 256
LEARNING_RATE = 0.0002
BETAS = (0.5, 0.999)
BATCH_SIZE = 32

# Define a new dataset class
class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = ImageFolder(root_dir, transform=transform)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx][0]

# Initialize the new Generator and Discriminator and move them to the GPU
G = Generator().to("cuda")
D = Discriminator().to("cuda")

# Load pretrained model state dictionaries
pretrained_path = "models/imagenet-256-state.pt"

G.load_state_dict(torch.load(pretrained_path)["G"], strict=False)
D.load_state_dict(torch.load(pretrained_path)["D"], strict=False)

# Define the dataset and data loader
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = MyDataset(root_dir="new_dataset", transform=transform)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Define the loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=BETAS)
optimizer_D = torch.optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=BETAS)

# TensorBoard writer
writer = SummaryWriter()

num_epochs = 500
latent_dim = 100

# Training loop
for epoch in range(num_epochs):
    total_real = 0
    correct_real = 0
    total_fake = 0
    correct_fake = 0

    for i, real_imgs in enumerate(data_loader):
        # Move real_imgs to the GPU
        real_imgs = real_imgs.to("cuda")

        batch_size = real_imgs.size(0)
        real_labels = torch.ones(batch_size, 1).to("cuda")
        fake_labels = torch.zeros(batch_size, 1).to("cuda")

        # ---------------------
        #  Train Discriminator
        # ---------------------

        D.zero_grad()

        # Real images
        real_outputs = D(real_imgs)
        d_real_loss = criterion(real_outputs, real_labels)

        # Fake images
        z = torch.randn(batch_size, latent_dim).to("cuda")
        fake_imgs = G(z)
        fake_outputs = D(fake_imgs.detach())
        d_fake_loss = criterion(fake_outputs, fake_labels)

        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Discriminator accuracy
        total_real += batch_size
        correct_real += (real_outputs > 0.5).sum().item()

        total_fake += batch_size
        correct_fake += (fake_outputs <= 0.5).sum().item()

        # ---------------------
        #  Train Generator
        # ---------------------

        G.zero_grad()

        # Generate fake images
        z = torch.randn(batch_size, latent_dim).to("cuda")
        fake_imgs = G(z)
        fake_outputs = D(fake_imgs)

        # Generator loss
        g_loss = criterion(fake_outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

        # Print progress
        if i % 100 == 0:
            print(
                f"[Epoch {epoch}/{num_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
            )

    # Save generated images at the end of each epoch
    if (epoch + 1) % 10 == 0:
        save_image(fake_imgs.data[:25], f"generated_images/epoch_{epoch + 1}.png", nrow=5, normalize=True)

    # Calculate and log discriminator accuracy
    accuracy_real = correct_real / total_real
    accuracy_fake = correct_fake / total_fake
    total_accuracy = (correct_real + correct_fake) / (total_real + total_fake)

    writer.add_scalar('Accuracy/Real', accuracy_real, epoch)
    writer.add_scalar('Accuracy/Fake', accuracy_fake, epoch)
    writer.add_scalar('Accuracy/Total', total_accuracy, epoch)
    writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch)
    writer.add_scalar('Loss/Generator', g_loss.item(), epoch)

# Close TensorBoard writer
writer.close()
