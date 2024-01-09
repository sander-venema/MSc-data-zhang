import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image

from model_new import Generator, Discriminator

# Define a new dataset class
class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = ImageFolder(root_dir, transform=transform)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx][0]

# Initialize the new Generator and Discriminator
G = Generator()
D = Discriminator()

# Define the dataset and data loader
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = MyDataset(root_dir="new_dataset", transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

# Define the loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

num_epochs = 200
latent_dim = 100

# Training loop
for epoch in range(num_epochs):
    for i, real_imgs in enumerate(data_loader):
        batch_size = real_imgs.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        D.zero_grad()

        # Real images
        real_outputs = D(real_imgs)
        d_real_loss = criterion(real_outputs, real_labels)

        # Fake images
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = G(z)
        fake_outputs = D(fake_imgs.detach())
        d_fake_loss = criterion(fake_outputs, fake_labels)

        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        #  Train Generator
        # ---------------------

        G.zero_grad()

        # Generate fake images
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = G(z)
        fake_outputs = D(fake_imgs)

        # Generator loss
        g_loss = criterion(fake_outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

        # Print progress
        if i % 100 == 0:
            print(
                f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(data_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
            )

    # Save generated images at the end of each epoch
    if (epoch + 1) % 10 == 0:
        save_image(fake_imgs.data[:25], f"generated_images/epoch_{epoch + 1}.png", nrow=5, normalize=True)
