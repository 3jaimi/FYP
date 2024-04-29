import os
import sys
from PIL import Image

def resize_images(input_dir, output_dir, offset):
    images = [f for f in os.listdir(input_dir) if f.endswith(".jpg") or f.endswith(".png")]
    for filename in images[offset:]:
        img = Image.open(os.path.join(input_dir, filename))
        img = img.resize((1024, 1024))
        img.save(os.path.join(output_dir, filename))

if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    offset = int(sys.argv[3])
    resize_images(input_dir, output_dir, offset)
