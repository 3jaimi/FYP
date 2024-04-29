import os
import shutil
import sys
import random
from PIL import Image

def get_min_files(root_dir):
    min_files = float('inf')
    for folder_name in os.listdir(root_dir):
        full_folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(full_folder_path):
            num_files = len(os.listdir(full_folder_path))
            if num_files < min_files:
                min_files = num_files
    return min_files

def crop_image(image_path):
    img = Image.open(image_path)
    width, height = img.size
    size = min(width, height)
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2
    img_cropped = img.crop((left, top, right, bottom))
    img_cropped.save(image_path)

def move_and_rename_images(root_dir, new_dir, num_files):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    for folder_name in os.listdir(root_dir):
        full_folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(full_folder_path):
            image_files = os.listdir(full_folder_path)
            selected_files = random.sample(image_files, num_files)
            for filename in selected_files:
                old_path = os.path.join(full_folder_path, filename)
                new_filename = f"{folder_name}-{filename}"
                new_path = os.path.join(new_dir, new_filename)

                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(new_path), exist_ok=True)

                # Move and rename
                shutil.move(old_path, new_path)

                # Crop image
                crop_image(new_path)

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py root_dir new_dir")
        sys.exit(1)

    root_dir = sys.argv[1]
    new_dir = sys.argv[2]
    num_files = get_min_files(root_dir)
    move_and_rename_images(root_dir, new_dir, num_files)

if __name__ == "__main__":
    main()