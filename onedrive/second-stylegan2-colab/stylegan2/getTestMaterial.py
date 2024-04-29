import os
import sys
from PIL import Image

def crop_images(input_path, output_dir):
    # Open the big image
    big_image = Image.open(input_path)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract dimensions of each small image
    width, height = big_image.size
    small_width = width // 7
    small_height = height // 4
    
    # Loop through each row and column to crop the small images
    for row in range(4):
        for col in range(7):
            # Calculate the coordinates for cropping
            left = col * small_width
            top = row * small_height
            right = left + small_width
            bottom = top + small_height
            
            # Crop the small image
            small_image = big_image.crop((left, top, right, bottom))
            
            # Save the cropped image with the desired name
            output_filename = f"{os.path.splitext(os.path.basename(input_path))[0]}-{row * 7 + col + 1}.png"
            output_path = os.path.join(output_dir, output_filename)
            small_image.save(output_path)
            print(f"Saved {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python crop_images.py <input_image_path> <output_directory>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    crop_images(input_path, output_dir)