import os
import shutil
from tqdm import tqdm

def list_directories_starting_with(directory, prefix):
    directories = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) and name.startswith(prefix)]
    return directories

ttv = ['train', 'test', 'val']
types = ['images', 'labels']
prefixes = ['BreaDM-Ma-', 'BreaDM-Be-']
# img_class = ['SUB2', 'VIBRANT', 'VIBRANT+C2']
img_class = ['VIBRANT', 'VIBRANT+C2', 'VIBRANT+C4']

directory_path = 'data/seg/'

train_dir = 'complete_dataset/train/'
test_dir = 'complete_dataset/test/'

new_directory = 'complete_dataset/'
os.makedirs(new_directory, exist_ok=True)

for split in ttv:
    print('Split: ' + split)
    if split == 'train' or split == 'val':
        new_directory = 'complete_dataset/train/'
    elif split == 'test':
        new_directory = 'complete_dataset/test/'
    for soort in types:
        print('Type: ' + soort)
        for prefix in prefixes:
            print('Prefix: ' + prefix)
            directory_path = 'data/seg/' + split + '/' + soort + '/'
            file_list = list_directories_starting_with(directory_path, prefix)
            img_dir = 'data/seg/' + split + '/' + soort + '/'

            for folder_name in tqdm(file_list):
                source_path = os.path.join(img_dir, folder_name)

                for img_class_name in img_class:
                    print('Class: ' + img_class_name)
                    destination_path = os.path.join(source_path, img_class_name)

                    # Copy the contents of the source folder to the 'VIBRANT' folder in the new directory and add a prefix to the file names
                    for filename in os.listdir(destination_path):
                        img = os.path.join(destination_path, filename)
                        fin_path = new_directory + soort + '/'
                        
                        os.makedirs(fin_path, exist_ok=True)
                        shutil.copy(img, fin_path)
                        os.rename(os.path.join(fin_path, filename), os.path.join(fin_path, folder_name + '_' + img_class_name + '_' + filename))
