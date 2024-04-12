import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

# Function to perform horizontal flip augmentation
def horizontal_flip(image):
    return np.fliplr(image)

# Define paths
base_path = 'new_dataset'
splits = ['train/', 'test/']

# Create directory for augmented dataset
output_base_path = 'augmented_dataset'
os.makedirs(output_base_path, exist_ok=True)

# Iterate through train and test splits
for split in splits:
    images_path = os.path.join(base_path, split + 'images/')
    masks_path = os.path.join(base_path, split + 'labels/')
    save_images_path = os.path.join(output_base_path, split + 'images/')
    save_masks_path = os.path.join(output_base_path, split + 'labels/')

    # Create directories for saving augmented images
    os.makedirs(save_images_path, exist_ok=True)
    os.makedirs(save_masks_path, exist_ok=True)

    # List images and masks in the current split
    image_files = os.listdir(images_path)
    masks_files = os.listdir(masks_path)

    # Perform augmentation for each image and mask
    for image_file, mask_file in tqdm(zip(image_files, masks_files), total=len(image_files)):
        # Load image
        image = cv2.imread(os.path.join(images_path, image_file))

        # Load mask
        mask = cv2.imread(os.path.join(masks_path, mask_file))

        # Perform horizontal flip augmentation
        augmented_image = horizontal_flip(image)
        augmented_mask = horizontal_flip(mask)

        # Save augmented images
        new_image_file = image_file.split('_')[0] + '_flip_' + image_file.split('_')[1]
        cv2.imwrite(os.path.join(save_images_path, new_image_file), augmented_image)

        # Save augmented masks
        new_mask_file = mask_file.split('_')[0] + '_flip_' + mask_file.split('_')[1]
        cv2.imwrite(os.path.join(save_masks_path, new_mask_file), augmented_mask)

    # Copy other files (e.g., additional data) if needed
    shutil.copytree(images_path, save_images_path, dirs_exist_ok=True)

print("Augmentation completed successfully!")
