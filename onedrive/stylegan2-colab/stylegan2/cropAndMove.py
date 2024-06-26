import os
import shutil
import sys
import random
from PIL import Image

def get_num_files(folder_name):
    if int(folder_name) == 140: 
        return 271
    elif int(folder_name) < 141:
        return 0
    else:
        return 297

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

def move_and_rename_images(root_dir, new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    for folder_name in sorted(os.listdir(root_dir)):
        full_folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(full_folder_path):
            image_files = [f for f in os.listdir(full_folder_path) if os.path.isfile(os.path.join(full_folder_path, f))]
            num_files = get_num_files(folder_name)
            selected_files = random.sample(image_files, min(num_files, len(image_files)))
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
    move_and_rename_images(root_dir, new_dir)

if __name__ == "__main__":
    main()
