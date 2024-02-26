import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from archs.model_wass import Generator, Discriminator, Discriminator_2
from utils.data_stuff import GenerationDataset
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Store training settings')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--loss', type=int, default=0, help='Loss function; 0: Wasserstein,')

args = parser.parse_args()

BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
IMAGE_SIZE = 512

filename = "wass_{0}_newmodel".format(LEARNING_RATE)
saving_path = "generated_images/{0}".format(filename)

os.makedirs(saving_path, exist_ok=True)

# Initialize the new Generator and Discriminator and move them to the GPU
G = Generator().to("cuda")
D = Discriminator_2().to("cuda")

# Load pretrained model state dictionaries
pretrained_path = "models/imagenet-512-state.pt"

G.load_state_dict(torch.load(pretrained_path)["G"], strict=False)
D.load_state_dict(torch.load(pretrained_path)["D"], strict=False)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = GenerationDataset(root_dir="new_dataset/train/images/", transform=transform)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Define the loss function and optimizers
# criterion = 
optimizer_G = torch.optim.RMSprop(G.parameters(), lr=LEARNING_RATE)
optimizer_D = torch.optim.RMSprop(D.parameters(), lr=LEARNING_RATE)

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

        D.zero_grad()

        # Real images
        print(real_imgs.shape)
        real_outputs = D(real_imgs)

        # Fake images
        z = torch.randn(batch_size, latent_dim).to("cuda")
        fake_imgs = G(z)
        fake_outputs = D(fake_imgs.detach())

        if (i + 1) % 5 == 0: # Train the discriminator every 5 batches
            d_loss = torch.mean(D(fake_imgs)) - torch.mean(D(real_imgs))
            d_loss.backward()
            optimizer_D.step()

            # Clip discriminator weights
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

        # Discriminator accuracy
        total_real += batch_size
        correct_real += (real_outputs > 0.5).sum().item()

        total_fake += batch_size
        correct_fake += (fake_outputs <= 0.5).sum().item()

        # ---------------------
        #  Train Generator
        # ---------------------

        # if (i + 1) % 5 == 0: # Train the generator every 5 batches
        G.zero_grad()

        # Generate fake images
        z = torch.randn(batch_size, latent_dim).to("cuda")
        fake_imgs = G(z)
        fake_outputs = D(fake_imgs)

        # Generator loss
        g_loss = -torch.mean(D(fake_imgs))
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
