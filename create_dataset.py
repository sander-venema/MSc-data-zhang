import os
import shutil
from tqdm import tqdm

def list_directories_starting_with(directory, prefix):
    directories = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) and name.startswith(prefix)]
    return directories

ttv = ['train', 'test', 'val']
types = ['images', 'labels']
prefixes = ['BreaDM-Ma-', 'BreaDM-Be-']

directory_path = 'data/seg/'

train_dir = 'new_dataset/train/'
test_dir = 'new_dataset/test/'

new_directory = 'new_dataset/'
os.makedirs(new_directory, exist_ok=True)

for split in ttv:
    print('Split: ' + split)
    if split == 'train' or split == 'val':
        new_directory = 'new_dataset/train/'
    elif split == 'test':
        new_directory = 'new_dataset/test/'
    for soort in types:
        print('Type: ' + soort)
        for prefix in prefixes:
            print('Prefix: ' + prefix)
            directory_path = 'data/seg/' + split + '/' + soort + '/'
            file_list = list_directories_starting_with(directory_path, prefix)
            img_dir = 'data/seg/' + split + '/' + soort + '/'

            for folder_name in tqdm(file_list):
                source_path = os.path.join(img_dir, folder_name)
                destination_path = os.path.join(source_path, 'VIBRANT')

                # Copy the contents of the source folder to the 'VIBRANT' folder in the new directory and add a prefix to the file names
                for filename in os.listdir(destination_path):
                    img = os.path.join(destination_path, filename)
                    if prefix == 'BreaDM-Ma-':
                        fin_path = new_directory + soort + '/' + 'malignant' + '/'
                    else:
                        fin_path = new_directory + soort + '/' + 'benign' + '/'
                    
                    os.makedirs(fin_path, exist_ok=True)
                    shutil.copy(img, fin_path)
                    os.rename(os.path.join(fin_path, filename), os.path.join(fin_path, folder_name + '_' + filename))
