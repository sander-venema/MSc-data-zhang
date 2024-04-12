import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
import random

# Function to perform combined augmentations
def apply_augmentations(image, mask):
    # Randomly decide whether to flip horizontally (50% probability)
    if random.random() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)

    # Random rotation within a range (e.g., ±10 degrees)
    rotation_angle = random.uniform(-10, 10)
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    mask = cv2.warpAffine(mask, rotation_matrix, (cols, rows))

    # Random scaling (uniformly sampled from 0.9 to 1.1)
    scaling_factor = random.uniform(0.9, 1.1)
    image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor)
    mask = cv2.resize(mask, None, fx=scaling_factor, fy=scaling_factor)

    return image, mask

# Define paths
base_path = 'new_dataset'
splits = ['train/', 'test/']

# Create directory for augmented dataset
output_base_path = 'augmented_dataset'
os.makedirs(output_base_path, exist_ok=True)

# Iterate through train and test splits
for split in splits:
    images_path = os.path.join(base_path, split + 'images/')
    labels_path = os.path.join(base_path, split + 'labels/')
    save_images_path = os.path.join(output_base_path, split + 'images/')
    save_labels_path = os.path.join(output_base_path, split + 'labels/')

    # Create directories for saving augmented images and labels
    os.makedirs(save_images_path, exist_ok=True)
    os.makedirs(save_labels_path, exist_ok=True)

    # List images and masks in the current split
    image_files = os.listdir(images_path)
    mask_files = os.listdir(labels_path)

    # Perform augmentation for each image and mask
    for image_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files)):
        # Load image
        image = cv2.imread(os.path.join(images_path, image_file))

        # Load corresponding label
        label = cv2.imread(os.path.join(labels_path, mask_file)) 

        # Apply augmentations to images and masks simultaneously
        augmented_image, augmented_label = apply_augmentations(image, label)

        # Save augmented images
        new_image_file = image_file.split('_')[0] + '_augmented_' + image_file.split('_')[1]
        cv2.imwrite(os.path.join(save_images_path, new_image_file), augmented_image)

        # Save augmented labels
        new_label_file = mask_file.split('_')[0] + '_augmented_' + mask_file.split('_')[1]
        cv2.imwrite(os.path.join(save_labels_path, new_label_file), augmented_label)

    shutil.copytree(images_path, save_images_path, dirs_exist_ok=True)
    shutil.copytree(labels_path, save_labels_path, dirs_exist_ok=True)

print("Augmentation completed successfully!")
