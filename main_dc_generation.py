import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from archs.model_dcgan import Generator, Discriminator
from utils.data_stuff import GenerationDataset
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Store training settings')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate')

args = parser.parse_args()

BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
IMAGE_SIZE = 512

filename = "dcgan_{0}".format(LEARNING_RATE)
saving_path = "generated_images/{0}".format(filename)

os.makedirs(saving_path, exist_ok=True)

G = Generator(channels=1).to("cuda")
D = Discriminator(channels=1).to("cuda")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = GenerationDataset(root_dir="new_dataset/train/images/", transform=transform)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Define the loss function and optimizers
criterion = torch.nn.BCELoss()  # Binary Cross-Entropy Loss for DCGAN
optimizer_G = torch.optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# TensorBoard writer
writer = SummaryWriter(f"logs_generation/{filename}")

num_epochs = 500
latent_dim = 100

# Training loop
for epoch in tqdm(range(num_epochs)):
    total_real = 0
    correct_real = 0
    total_fake = 0
    correct_fake = 0

    for i, real_imgs in enumerate(data_loader):
        # Move real_imgs to the GPU
        real_imgs = real_imgs.to("cuda")

        batch_size = real_imgs.size(0)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        z = torch.randn(batch_size, latent_dim, 1, 1).to("cuda")
        real_labels = torch.ones(batch_size).to("cuda")
        fake_labels = torch.zeros(batch_size).to("cuda")

        outputs = D(real_imgs)
        # print(real_imgs.shape)
        # print(outputs.shape, real_labels.shape)
        d_loss_real = criterion(outputs.flatten(), real_labels)
        real_score = outputs

        fake_imgs = G(z)
        outputs = D(fake_imgs)
        d_loss_fake = criterion(outputs.flatten(), fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Discriminator accuracy
        total_real += batch_size
        correct_real += (real_score > 0.5).sum().item()

        total_fake += batch_size
        correct_fake += (fake_score <= 0.5).sum().item()

        # ---------------------
        #  Train Generator
        # ---------------------

        z = torch.randn(batch_size, latent_dim, 1, 1).to("cuda")
        fake_imgs = G(z)
        outputs = D(fake_imgs)
        g_loss = criterion(outputs.flatten(), real_labels)

        G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    # Save generated images at the end of each epoch
    if (epoch + 1) % 10 == 0:
        save_image(fake_imgs.data[:25], f"{saving_path}/epoch_{epoch + 1}.png", nrow=5, normalize=True)

    # Calculate and log discriminator accuracy
    accuracy_real = correct_real / total_real
    accuracy_fake = correct_fake / total_fake

    writer.add_scalar('Accuracy/Real', accuracy_real, epoch)
    writer.add_scalar('Accuracy/Fake', accuracy_fake, epoch)
    writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch)
    writer.add_scalar('Loss/Generator', g_loss.item(), epoch)

# Close TensorBoard writer
writer.close()

# Save the model state dictionaries
torch.save({
    "G": G.state_dict(),
    "D": D.state_dict()
}, f"saved_models/{filename}.pt")
