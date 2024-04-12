import os

def get_folder_names(directory):
    """
    Returns a list of folder names inside the specified directory.
    """
    return [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

def compare_folders(train_dir, test_dir):
    """
    Compares folder names in train and test directories and prints common folder names.
    """
    train_folders = get_folder_names(train_dir)
    test_folders = get_folder_names(test_dir)

    common_folders = set(train_folders) & set(test_folders)

    if common_folders:
        print("Common folder names between train and test sets:")
        for folder in common_folders:
            print(folder)
    else:
        print("No common folder names found between train and test sets.")

# Specify your dataset paths
train_images_dir = 'data/seg/train/images'
test_images_dir = 'data/seg/test/images'

compare_folders(train_images_dir, test_images_dir)
