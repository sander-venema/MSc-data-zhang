import os
import shutil

def list_directories_starting_with(directory, prefix):
    directories = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) and name.startswith(prefix)]
    return directories

# Replace 'your_directory_path' with the actual path to the directory you want to search in
directory_path = 'data/seg3D/train/images/'
prefix_to_search = 'BreaDM-Ma-'

file_list = list_directories_starting_with(directory_path, prefix_to_search)

img_dir = 'data/seg/train/images/'

# Create a new directory to store the images
new_directory = 'new_dataset/images/'
os.makedirs(new_directory, exist_ok=True)

for folder_name in file_list:
    source_path = os.path.join(img_dir, folder_name)
    destination_path = os.path.join(source_path, 'VIBRANT')
    print(destination_path)

    # Copy the contents of the source folder to the 'VIBRANT' folder in the new directory and add a prefix to the file names
    for filename in os.listdir(destination_path):
        img = os.path.join(destination_path, filename)
        print(img)
        print(new_directory)
        shutil.copy(img, new_directory)
        os.rename(os.path.join(new_directory, filename), os.path.join(new_directory, folder_name + '_' + filename))

directory_path = 'data/seg/train/labels/'
prefix_to_search = 'BreaDM-Ma-'

file_list = list_directories_starting_with(directory_path, prefix_to_search)

img_dir = 'data/seg/train/labels/'

# Create a new directory to store the images
new_directory = 'new_dataset/labels/'
os.makedirs(new_directory, exist_ok=True)

for folder_name in file_list:
    source_path = os.path.join(img_dir, folder_name)
    destination_path = os.path.join(source_path, 'VIBRANT')
    print(destination_path)

    for filename in os.listdir(destination_path):
        img = os.path.join(destination_path, filename)
        print(img)
        print(new_directory)
        shutil.copy(img, new_directory)
        os.rename(os.path.join(new_directory, filename), os.path.join(new_directory, folder_name + '_' + filename))
