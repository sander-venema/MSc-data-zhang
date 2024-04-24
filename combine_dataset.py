import os
import shutil
import random

# Define paths
dataset_path = "new_dataset"
combined_path = "combined_dataset"

# Function to combine subfolders
def combine_subfolders(source_path, dest_path):
    os.makedirs(dest_path, exist_ok=True)
    for file_name in os.listdir(source_path):
        src_file = os.path.join(source_path, file_name)
        dest_file = os.path.join(dest_path, file_name)
        shutil.copy(src_file, dest_file)

# Combine images
combine_subfolders(os.path.join(dataset_path, "train", "images"), os.path.join(combined_path, "images"))
combine_subfolders(os.path.join(dataset_path, "test", "images"), os.path.join(combined_path, "images"))

# Combine labels
combine_subfolders(os.path.join(dataset_path, "train", "labels"), os.path.join(combined_path, "labels"))
combine_subfolders(os.path.join(dataset_path, "test", "labels"), os.path.join(combined_path, "labels"))

# Split combined dataset into train and test
combined_images_path = os.path.join(combined_path, "images")
combined_labels_path = os.path.join(combined_path, "labels")

train_images_path = os.path.join(combined_path, "train", "images")
train_labels_path = os.path.join(combined_path, "train", "labels")
test_images_path = os.path.join(combined_path, "test", "images")
test_labels_path = os.path.join(combined_path, "test", "labels")

os.makedirs(train_images_path, exist_ok=True)
os.makedirs(train_labels_path, exist_ok=True)
os.makedirs(test_images_path, exist_ok=True)
os.makedirs(test_labels_path, exist_ok=True)

# Get all image and label files
image_files = os.listdir(combined_images_path)
label_files = os.listdir(combined_labels_path)

# Shuffle the files
combined_files = list(zip(image_files, label_files))
random.shuffle(combined_files)
image_files, label_files = zip(*combined_files)

# Split the files into train and test
train_split = int(len(image_files) * 0.8)  # 80% for training
train_images = image_files[:train_split]
train_labels = label_files[:train_split]
test_images = image_files[train_split:]
test_labels = label_files[train_split:]

# Move train images and labels to train directories
for image_file, label_file in zip(train_images, train_labels):
    src_image = os.path.join(combined_images_path, image_file)
    dest_image = os.path.join(train_images_path, image_file)
    shutil.move(src_image, dest_image)
    src_label = os.path.join(combined_labels_path, label_file)
    dest_label = os.path.join(train_labels_path, label_file)
    shutil.move(src_label, dest_label)

# Move test images and labels to test directories
for image_file, label_file in zip(test_images, test_labels):
    src_image = os.path.join(combined_images_path, image_file)
    dest_image = os.path.join(test_images_path, image_file)
    shutil.move(src_image, dest_image)
    src_label = os.path.join(combined_labels_path, label_file)
    dest_label = os.path.join(test_labels_path, label_file)
    shutil.move(src_label, dest_label)
