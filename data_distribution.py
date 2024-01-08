import numpy as np
from matplotlib import pyplot as plt
import os

directories = []

def list_directories_starting_with(directory, prefix):
    directories = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) and name.startswith(prefix)]
    return directories

# Replace 'your_directory_path' with the actual path to the directory you want to search in
directory_path = 'data/seg3D/train/images/'
prefix_to_search = 'BreaDM-Ma-'

file_list = list_directories_starting_with(directory_path, prefix_to_search)

img_dir = 'data/seg3D/train/images/'
mask_dir = 'data/seg3D/train/labels/'
for k in range(len(file_list)):
    # Load data
    img_array = np.load(img_dir + file_list[k] + '/VIBRANT.npy')
    mask_array = np.load(mask_dir + file_list[k] + '/VIBRANT.npy')

    hist, bins = np.histogram(img_array, bins=256)

    plt.figure()
    plt.title('Histogram of image intensities')
    plt.xlabel('Grayscale value')
    plt.ylabel('Pixel count')
    plt.xlim([0.0, 1.0])
    plt.ylim([0, 1000000])
    plt.plot(bins[:-1], hist)

    plt.savefig('dist/malignant_VIBRANT/histogram_{}.png'.format(file_list[k]))
