import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, img_transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.images_path = os.path.join(self.root_dir, "images")
        self.masks_path = os.path.join(self.root_dir, "labels")
        self.image_list = os.listdir(self.images_path)
        self.mask_list = os.listdir(self.masks_path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_path, self.image_list[idx])
        mask_name = os.path.join(self.masks_path, self.mask_list[idx])

        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")

        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask
    
class GenerationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name).convert("L")

        if self.transform:
            image = self.transform(image)

        return image
    
image_transforms = transforms.Compose([
    transforms.Resize((369, 369)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transforms = transforms.Compose([
    transforms.Resize((369, 369)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.round(x)),
])
