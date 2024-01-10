from torch import nn

IMAGE_SIZE = 256

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, IMAGE_SIZE, IMAGE_SIZE)):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, IMAGE_SIZE),
            nn.BatchNorm1d(IMAGE_SIZE),
            nn.ReLU(),
            nn.Linear(IMAGE_SIZE, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1 * IMAGE_SIZE * IMAGE_SIZE),  # Adjust output size to match the image size
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, IMAGE_SIZE, IMAGE_SIZE)  # Reshape to the image size
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, IMAGE_SIZE, IMAGE_SIZE)):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1 * IMAGE_SIZE * IMAGE_SIZE, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, IMAGE_SIZE),
            nn.LeakyReLU(0.2),
            nn.Linear(IMAGE_SIZE, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity