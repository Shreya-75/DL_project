import os
import cv2
import numpy as np
from tqdm import tqdm
import random
import shutil

# Constants
SOURCE_DIR = 'data'  # Directory containing the original dataset
TARGET_DIR = 'augmented_data1'  # Directory to save augmented images
TARGET_IMAGE_COUNT = 5000  # Total number of images per folder
TRAIN_RATIO = 0.7  # 70% for training
VAL_RATIO = 0.2    # 20% for validation
TEST_RATIO = 0.1   # 10% for testing

# Create target directory
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)  # Create main target directory if it does not exist

# Create subdirectories for train, val, and test
for subdir in ['train', 'val', 'test']:
    subdir_path = os.path.join(TARGET_DIR, subdir)
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)

# Function to perform data augmentation
def augment_image(image):
    # Randomly apply a series of transformations
    # 1. Brightness adjustment
    brightness_variation = np.random.uniform(0.7, 1.3)  # Variation factor
    image = cv2.convertScaleAbs(image, alpha=brightness_variation, beta=0)  # Adjust brightness

    # 2. Color variation (Hue)
    color_variation = np.random.uniform(0.7, 1.3)  # Variation factor
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[..., 0] = np.clip(image[..., 0] + np.random.randint(-20, 20), 0, 180)  # Adjust hue
    image[..., 1] = np.clip(image[..., 1] * color_variation, 0, 255)  # Adjust saturation
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    # 3. Random flipping
    if np.random.rand() > 0.5:  # 50% chance to flip
        image = cv2.flip(image, 1)  # Horizontal flip

    # 4. Scaling
    scale_factor = np.random.uniform(0.8, 1.2)  # Scale factor
    new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    image = cv2.resize(image, new_size)
    # Crop back to original size if scaled up
    if new_size[0] > image.shape[1] or new_size[1] > image.shape[0]:
        x_start = (new_size[0] - image.shape[1]) // 2
        y_start = (new_size[1] - image.shape[0]) // 2
        image = image[y_start:y_start+image.shape[0], x_start:x_start+image.shape[1]]

    # 5. Contrast adjustment
    contrast_factor = np.random.uniform(0.7, 1.3)  # Contrast factor
    image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)  # Adjust contrast

    # 6. Add noise
    if np.random.rand() > 0.5:  # 50% chance to add noise
        noise = np.random.normal(0, 25, image.shape)
        noise = noise.astype(np.uint8)
        image = cv2.add(image, noise)

    return image

# Process each folder and perform augmentation
for folder in os.listdir(SOURCE_DIR):
    folder_path = os.path.join(SOURCE_DIR, folder)

    if os.path.isdir(folder_path):
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'png', 'jpeg'))]
        total_images = len(image_files)

        # Create the same subfolder in the target directory
        for subfolder in ['train', 'val', 'test']:
            subfolder_path = os.path.join(TARGET_DIR, subfolder, folder)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)  # Create subfolder in train/val/test if it does not exist

        # Augment images to reach exactly TARGET_IMAGE_COUNT per sub-dataset
        augmented_images = []
        for _ in tqdm(range(TARGET_IMAGE_COUNT), desc=f"Augmenting {folder}"):
            image_file = image_files[_ % total_images]  # Loop through existing images
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)

            # Augment the image
            augmented_image = augment_image(image)

            augmented_images.append(augmented_image)

        # Split the augmented images into train, val, and test sets
        num_train = int(TARGET_IMAGE_COUNT * TRAIN_RATIO)
        num_val = int(TARGET_IMAGE_COUNT * VAL_RATIO)
        num_test = TARGET_IMAGE_COUNT - num_train - num_val

        # Shuffle augmented images to randomize
        random.shuffle(augmented_images)

        # Distribute images into train, val, and test folders
        for i, augmented_image in enumerate(augmented_images):
            if i < num_train:
                target_subfolder = 'train'
            elif i < num_train + num_val:
                target_subfolder = 'val'
            else:
                target_subfolder = 'test'

            target_folder = os.path.join(TARGET_DIR, target_subfolder, folder)
            target_image_path = os.path.join(target_folder, f"{i}.jpg")
            cv2.imwrite(target_image_path, augmented_image)

print("Data augmentation and dataset splitting completed!")
