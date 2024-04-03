import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from sklearn.model_selection import KFold
from archs.model_wass import Generator_3, Discriminator_3
from utils.data_stuff import GenerationDataset
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Store training settings')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--latent_dim', type=int, default=100, help='Latent dimension')
parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs')
parser.add_argument('--lamb', type=int, default=0.1, help='Lambda value for gradient penalty')
parser.add_argument('--complete_dataset', action='store_true', default=False, help='Whether to use the complete dataset')
parser.add_argument('--augmented_dataset', action='store_true', default=False, help='Whether to use the augmented dataset')

args = parser.parse_args()

BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
latent_dim = args.latent_dim
num_epochs = args.num_epochs
LAMBDA = args.lamb

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Normalize((0.5,), (0.5,))
])

def calc_gradient_penalty(netD, real_data, fake_data, batch_size, dim, device, gp_lambda):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 1, dim, dim)
    alpha = alpha.to(device)
    
    fake_data = fake_data.view(batch_size, 1, dim, dim)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty

wasserstein_loss = lambda y_true, y_pred: -torch.mean(y_true * y_pred)

if args.complete_dataset:
    dataset = GenerationDataset(root_dir="complete_dataset/train/images/", transform=transform)
    print("Using complete dataset")
elif args.augmented_dataset:
    dataset = GenerationDataset(root_dir="augmented_dataset/train/images/", transform=transform)
    print("Using augmented dataset")
else:
    dataset = GenerationDataset(root_dir="new_dataset/train/images/", transform=transform)
    print("Using original dataset")

# Use KFold for 3-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

one = torch.tensor(1, dtype=torch.float)
mone = one * -1
one.to("cuda")
mone.to("cuda")

for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
    G = Generator_3().to("cuda")
    D = Discriminator_3().to("cuda")

    G.load_state_dict(torch.load("models/imagenet-512-state.pt")["G"], strict=False)
    D.load_state_dict(torch.load("models/imagenet-512-state.pt")["D"], strict=False)

    optimizer_G = torch.optim.RMSprop(G.parameters(), lr=LEARNING_RATE, weight_decay=2e-5)
    optimizer_D = torch.optim.RMSprop(D.parameters(), lr=LEARNING_RATE, weight_decay=2e-5) 

    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    filename = "wassGPaug{0}_{1}_5fold_{2}".format(BATCH_SIZE, LEARNING_RATE, fold)
    saving_path = "generated_images/{0}".format(filename)
    writer = SummaryWriter(f"logs_generation/{filename}")
    os.makedirs(saving_path, exist_ok=True)

    cur_best_real = 0
    cur_best_fake = 0

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        total_real = 0
        correct_real = 0
        total_fake = 0
        correct_fake = 0
        real_score = 0
        fake_score = 0

        for i, real_imgs in enumerate(train_loader):
            G.train()
            D.train()
            real_imgs = real_imgs.to("cuda")
            batch_size = real_imgs.size(0)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            D.zero_grad()

            # Real images
            real_outputs = D(real_imgs)
            real_score += real_outputs.mean().item()
            d_loss_real = real_outputs.mean()

            # Fake images
            z = torch.randn(batch_size, latent_dim).to("cuda")
            fake_imgs = G(z)
            fake_outputs = D(fake_imgs.detach())
            fake_score += fake_outputs.mean().item()
            d_loss_fake = fake_outputs.mean()

            gradient_penalty = calc_gradient_penalty(D, real_imgs.data, fake_imgs.data, batch_size, 369, "cuda", LAMBDA)

            d_loss = -torch.mean(real_outputs) + torch.mean(fake_outputs) + gradient_penalty
            # d_loss = d_loss_fake - d_loss_real + gradient_penalty
            d_loss.backward()
            optimizer_D.step()

            # Clip discriminator weights
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

            # Discriminator accuracy
            total_real += batch_size
            correct_real += (torch.sigmoid(real_outputs) >= 0.5).sum().item()

            total_fake += batch_size
            correct_fake += (torch.sigmoid(fake_outputs) < 0.5).sum().item()

            # ---------------------
            #  Train Generator
            # ---------------------

            if (i + 1) % 5 == 0: # Train the generator every 5 batches
                G.zero_grad()

                # Generate fake images
                z = torch.randn(batch_size, latent_dim).to("cuda")
                fake_imgs = G(z)
                fake_outputs = D(fake_imgs)

                # Generator loss
                g_loss = torch.mean(D(fake_imgs))
                g_loss.backward(mone)
                optimizer_G.step()
                g_loss = -g_loss

        accuracy_real = correct_real / total_real
        accuracy_fake = correct_fake / total_fake

        real_score /= len(train_loader)
        fake_score /= len(train_loader)

        writer.add_scalar('Accuracy/Real', accuracy_real, epoch)
        writer.add_scalar('Accuracy/Fake', accuracy_fake, epoch)
        writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch)
        writer.add_scalar('Loss/Generator', g_loss.item(), epoch)
        writer.add_scalar('Score/Real', real_score, epoch)
        writer.add_scalar('Score/Fake', fake_score, epoch)
        
        G.eval()
        D.eval()

        val_loss = 0
        val_acc_real = 0
        val_acc_fake = 0
        num_val_batches = 0

        for val_imgs in val_loader:
            val_imgs = val_imgs.to("cuda")
            batch_size = val_imgs.size(0)

            # Calculate discriminator outputs for real and fake images
            real_outputs = D(val_imgs)
            z = torch.randn(batch_size, latent_dim).to("cuda")
            fake_imgs = G(z)
            fake_outputs = D(fake_imgs.detach())

            # Calculate discriminator loss and accuracy
            d_loss = torch.mean(D(fake_imgs)) - torch.mean(D(val_imgs))
            val_loss += d_loss.item()
            val_acc_real += (torch.sigmoid(real_outputs) >= 0.5).sum().item()
            val_acc_fake += (torch.sigmoid(fake_outputs) < 0.5).sum().item()

            num_val_batches += 1

        # Calculate average loss and accuracy
        val_loss /= num_val_batches
        val_acc_real /= len(val_dataset)
        val_acc_fake /= len(val_dataset)

        # Log validation loss and accuracy
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/Accuracy/Real', val_acc_real, epoch)
        writer.add_scalar('Validation/Accuracy/Fake', val_acc_fake, epoch)

        # Save the best model
        if val_acc_real > 0.8 and val_acc_real >= cur_best_real and val_acc_fake > 0.8 and val_acc_fake >= cur_best_fake:
            with open(f"saved_models/{filename}_best.txt", "a+") as f:
                f.write(f"Epoch: {epoch}, Real accuracy: {val_acc_real}, Fake accuracy: {val_acc_fake}\n")
            cur_best_real = val_acc_real
            cur_best_fake = val_acc_fake
            torch.save({
                "G": G.state_dict(),
                "D": D.state_dict()
            }, f"saved_models/{filename}_best.pt")

        # Save generated images of the current state
        if (epoch+1) % 10 == 0:
            current_fake_imgs = G(torch.randn(16, latent_dim).to("cuda"))
            save_image(current_fake_imgs, f"{saving_path}/fold_{fold}_epoch_{epoch}.png", nrow=4, normalize=True)

    # # Save the model state dictionaries
    # torch.save({
    #     "G": G.state_dict(),
    #     "D": D.state_dict()
    # }, f"saved_models/{filename}.pt")

# Close TensorBoard writer
writer.close()