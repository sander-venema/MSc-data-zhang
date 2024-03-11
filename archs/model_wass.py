from torch import nn
import numpy as np
import torch
# torch.backends.cudnn.enabled = False

img_shape = (1, 369, 369)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *block(100, 128, normalize=False),
            *block(128, 256),
            # nn.Dropout(0.5),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            # nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img

class Generator_2(nn.Module):
    def __init__(self):
        super(Generator_2, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        img = self.model(z)
        return img
    
class Generator_3(nn.Module):
    def __init__(self, width=512):
        super(Generator_3, self).__init__()

        self.dim_z = 100
        self.module = nn.Sequential(
            nn.Linear(self.dim_z, width), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Linear(width, 2 * width), nn.BatchNorm1d(2 * width), nn.ReLU(),
            nn.Linear(2 * width, 2 * width), nn.BatchNorm1d(2 * width), nn.ReLU(),
            nn.Linear(2 * width, 2 * width), nn.BatchNorm1d(2 * width), nn.ReLU(),
            nn.Linear(2 * width, width), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Linear(width, 369 ** 2), nn.Tanh()
        )

    def forward(self, z):
        img = self.module(z)
        img = img.view(img.shape[0], *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(369 ** 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

class Discriminator_2(nn.Module):
    def __init__(self):
        super(Discriminator_2, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 1, 4, 1, 0),
        )
        
    def forward(self, img):
        x = self.model(img)
        return torch.flatten(x)
    
class Discriminator_3(nn.Module):
    def __init__(self, width=512, return_probs=True):
        super(Discriminator_3, self).__init__()

        self.module = nn.Sequential(
            nn.Linear(369 ** 2, width), nn.ReLU(),
            nn.Linear(width, 2 * width), nn.ReLU(),
            nn.Linear(2 * width, 2 * width), nn.ReLU(),
            nn.Linear(2 * width, width), nn.ReLU(),
            nn.Linear(width, 1)
            # ,nn.Sigmoid() if return_probs else nn.Sequential(),
        )

    def forward(self, z):
        img_flat = z.view(z.shape[0], -1)
        return self.module(img_flat)