import numpy as np
from matplotlib import pyplot as plt
import os

def list_directories_starting_with(directory, prefix):
    directories = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) and name.startswith(prefix)]
    return directories

# Replace 'your_directory_path' with the actual path to the directory you want to search in
directory_path = 'data/seg3D/train/images/'
prefix_to_search = 'BreaDM-Be-'

file_list = list_directories_starting_with(directory_path, prefix_to_search)

img_dir = 'data/seg3D/train/images/'
mask_dir = 'data/seg3D/train/labels/'
for k in range(len(file_list)):
    # Load data
    img_array = np.load(img_dir + file_list[k] + '/VIBRANT.npy')
    mask_array = np.load(mask_dir + file_list[k] + '/VIBRANT.npy')

    # Plot all slices along the third dimension
    # num_slices = img_array.shape[2]

    # fig, axes = plt.subplots(1, num_slices, figsize=(15, 3))

    # for i in range(num_slices):
    #     axes[i].imshow(img_array[:, :, i], cmap='gray')
    #     axes[i].set_title('Slice {}'.format(i))

    # plt.show()

    # # Plot all slices along the third dimension
    # num_slices = mask_array.shape[2]

    # fig, axes = plt.subplots(1, num_slices, figsize=(15, 3))

    # for i in range(num_slices):
    #     axes[i].imshow(mask_array[:, :, i], cmap='gray')
    #     axes[i].set_title('Slice {}'.format(i))

    # plt.show()  

    # Stack masks on images
    img_array = np.expand_dims(img_array, axis=3)
    mask_array = np.expand_dims(mask_array, axis=3)
    stacked_array = np.concatenate((img_array, mask_array), axis=3)

    # Plot all slices along the third dimension
    num_slices = stacked_array.shape[2]

    fig, axes = plt.subplots(1, num_slices-1, figsize=(15, 3))

    for i in range(num_slices-1):
        axes[i].imshow(stacked_array[:, :, i, 0], cmap='gray')
        axes[i].imshow(stacked_array[:, :, i, 1], cmap='jet', alpha=0.5)
        axes[i].set_title('Slice {}'.format(i))

    plt.savefig('img/benign_VIBRANT/stacked_masks_{}.png'.format(file_list[k]))
