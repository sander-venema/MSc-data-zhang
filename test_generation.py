import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from archs.model_wass import Generator_3, Discriminator_3
from utils.data_stuff import GenerationDataset
from tqdm import tqdm

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Normalize((0.5,), (0.5,))
])

G = Generator_3().to("cuda")
D = Discriminator_3().to("cuda")

models_dir = "saved_models/generation/"
models = os.listdir(models_dir)

count = 0
for model in models:
    print(f"Testing model {count + 1}/{len(models)}")
    count += 1
    pretrained_gan = os.path.join(models_dir, model)
    G.load_state_dict(torch.load(pretrained_gan)["G"], strict=False)
    D.load_state_dict(torch.load(pretrained_gan)["D"], strict=False)

    gan_dataset = GenerationDataset(root_dir="flip_dataset/test/images/", transform=transform)
    gan_loader = DataLoader(gan_dataset, batch_size=1, shuffle=False)

    # Define the testing loop
    G.eval()
    D.eval()
    real_correct = 0
    fake_correct = 0
    total = 0

    with torch.no_grad():
        for i, real_images in enumerate(tqdm(gan_loader)):
            real_images = real_images.to("cuda")
            z = torch.randn(1, 100).to("cuda")
            fake_images = G(z).to("cuda")

            # Compute accuracy for real images
            real_outputs = D(real_images)
            real_predictions = torch.round(torch.sigmoid(real_outputs))
            real_correct += torch.sum(real_predictions == 1).item()

            # Compute accuracy for fake images
            fake_outputs = D(fake_images)
            fake_predictions = torch.round(torch.sigmoid(fake_outputs))
            fake_correct += torch.sum(fake_predictions == 0).item()

            total += real_images.size(0)

    # Compute accuracy
    real_accuracy = real_correct / total
    fake_accuracy = fake_correct / total

    # Write accuracy to a txt file
    with open("accuracy_gen.txt", "a+") as f:
        f.write(f"{model}: ")
        f.write(f"Real Accuracy: {real_accuracy}, ")
        f.write(f"Fake Accuracy: {fake_accuracy}\n")