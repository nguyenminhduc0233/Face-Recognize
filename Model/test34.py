import os
import cv2
import random
import shutil
from sklearn.model_selection import train_test_split
import numpy as np

def augment_images_in_folders(parent_folder):
    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.isdir(folder_path):
            image_count = len(os.listdir(folder_path))
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                original_image = cv2.imread(image_path)
                flipped_image = cv2.flip(original_image, 1)
                cv2.imwrite(os.path.join(folder_path, f'flipped_horizontal_{image_name}'), flipped_image)

            if image_count *2 < 5:
                augmentation_factor = np.ceil(5 / (image_count * 2))
                for image_name in os.listdir(folder_path):
                    original_image_path = os.path.join(folder_path, image_name)
                    original_image = cv2.imread(original_image_path)
                    for i in range(int(augmentation_factor)):
                        # Random rotation angle between -5 and 5 degrees
                        rotation_angle = random.uniform(-15, 15)
                        rotated_image = rotate_image(original_image, rotation_angle)
                        cv2.imwrite(os.path.join(folder_path, f'rotated_{i}_{image_name}'), rotated_image)

            if True:
                augmentation_factor = 1
                for image_name in os.listdir(folder_path):
                    original_image_path = os.path.join(folder_path, image_name)
                    original_image = cv2.imread(original_image_path)
                    for i in range(int(augmentation_factor)):
                        # Random rotation angle between -5 and 5 degrees
                        rotation_angle = random.uniform(-15, 15)
                        rotated_image = rotate_image(original_image, rotation_angle)
                        cv2.imwrite(os.path.join(folder_path, f'rotated_{i}_{image_name}'), rotated_image)

def rotate_image(image, angle):
    # Calculate the rotation matrix
    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1.0)  # Sử dụng tham số M để chia nhỏ hơn 1 độ
    # Perform the affine transformation
    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    return rotated_image

def split_images(parent_dir, train_dir, test_dir, val_dir, train_size, test_size, val_size):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    sub_dirs = [os.path.join(parent_dir, name) for name in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, name))]
    
    for dir in sub_dirs:
        images = [os.path.join(dir, img) for img in os.listdir(dir) if os.path.isfile(os.path.join(dir, img))]
        
        train_images, temp_images = train_test_split(images, test_size=(1 - train_size))
        val_images, test_images = train_test_split(temp_images, test_size=(test_size / (test_size + val_size)))
        
        for img in train_images:
            new_path = os.path.join(train_dir, os.path.relpath(img, parent_dir))
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.move(img, new_path)
        
        for img in test_images:
            new_path = os.path.join(test_dir, os.path.relpath(img, parent_dir))
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.move(img, new_path)
        
        for img in val_images:
            new_path = os.path.join(val_dir, os.path.relpath(img, parent_dir))
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.move(img, new_path)
# Sử dụng hàm
parent_folder = r"extra_data"
augment_images_in_folders(parent_folder)
# Đường dẫn đến thư mục chứa tất cả các thư mục con (tương ứng với các nhãn)
main_directory = r"data\extra_data - Copy"
train_directory = r"data\extra_data_train"
validation_directory = r"data\extra_data_val"
test_directory = r"data\extra_data_test"

train_size = 0.6
val_size = 0.3
test_size = 0.1
split_images(main_directory, train_directory, test_directory, validation_directory, train_size, test_size, val_size)
