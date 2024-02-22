import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from backbones_unet.model.unet import Unet
from backbones_unet.utils.dataset import SemanticSegmentationDataset

from archs.model_wass import Generator, Discriminator
from utils.data_stuff import GenerationDataset
from tqdm import tqdm

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Normalize((0.5,), (0.5,))
])

G = Generator().to("cuda")
D = Discriminator().to("cuda")

pretrained_gan = "saved_models/wass_512_5e-05_drop.pt"
G.load_state_dict(torch.load(pretrained_gan)["G"], strict=False)
D.load_state_dict(torch.load(pretrained_gan)["D"], strict=False)

model = Unet(
    backbone='vgg16_bn',
    in_channels=3,
    num_classes=1,
)

pretrained_segmentation = "saved_models/segmentation/unet_vgg16bn_dice_bce_0.0001_new.pth"
model.load_state_dict(torch.load(pretrained_segmentation))
model.to("cuda")
model.eval()

seg_dataset = SemanticSegmentationDataset("new_dataset/test/images", "new_dataset/test/labels")
seg_loader = DataLoader(seg_dataset, batch_size=1, shuffle=False)

gan_dataset = GenerationDataset(root_dir="new_dataset/test/images/", transform=transform)
gan_loader = DataLoader(gan_dataset, batch_size=1, shuffle=False)
it = iter(gan_loader)

for i, (images, masks) in enumerate(seg_loader):
    images = images.to("cuda")
    masks = masks.to("cuda")

    # Get first image from gan_loader
    img = next(it).to("cuda")

    # With odds of 25%, generate a new image
    if torch.rand(1) < 0.25:
        G.eval()
        z = torch.randn(1, 100).to("cuda")
        img = G(z)
        print("Generated new image")

    # Let discriminator decide if the image is real or fake, skip if it believes it's fake
    confidence = D(img).detach().cpu().numpy()
    print(confidence[0])

    # If the discriminator is not confident, then we can use the image
    if confidence[0] < 0.5:
        # with torch.no_grad():
        #     outputs = model(images)
        #     outputs = torch.sigmoid(outputs)
        #     outputs = (outputs > 0.5).float()

        # image = images.cpu().numpy()
        # mask = outputs.cpu().numpy()
        # print("Segmentation done")
        pass
    else:
        print("Discriminator was not confident, skipping segmentation")
