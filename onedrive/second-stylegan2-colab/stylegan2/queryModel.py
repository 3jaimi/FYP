import argparse
import tensorflow as tf
import dnnlib.tflib as tflib
import pickle
from dnnlib import EasyDict
import os
import glob
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import tensorflow.keras as tensorKeras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dnnlib.tflib.autosummary import autosummary
from training import misc
from PIL import Image
import time


def load_stylegan_networks(pkl_file):
    tflib.init_tf()
    print('Loading networks from "%s"...' % pkl_file)
    with open(pkl_file, 'rb') as f:
        _G, _D, Gs = pickle.load(f)

    return _G, _D, Gs

def queryDiscriminator(model, img):
    image = Image.open(img)
    h,w=image.size
    if (h !=1024 or w!=1024):
        image=image.resize((1024, 1024), Image.ANTIALIAS)
    image = np.transpose(np.asarray(image), (2,0,1))[None]
    labels = np.array([]).reshape(1,0)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels)
    images = misc.adjust_dynamic_range(image, [0, 255], [-1, 1])
    score = model.get_output_for(images, labels,is_training=False)
    #probability = tf.math.softplus(score)
    parsedScore=score.eval()[0][0]
    if (parsedScore>-5):
        return "THIS IMAGE IS REAL"
    else:
        return "THIS IMAGE IS NOT REAL"

def watchDir(dir):
    while True:
        if (len(os.listdir(dir))>0):
            return os.path.join(dir,os.listdir(dir)[0])

def write_to_file(file_path, content, image_path):
    try:
        with open(file_path, "w") as file:
            file.write(content)
        print(f"Content successfully written to {file_path}")
        time.sleep(90)
        os.remove(file_path)
        print(f"Successfully deleted the file")
        os.remove(image_path)
        print(f"Successfully deleted the image")
    except Exception as e:
        print(f"Error writing to file: {e}")

def main():   
    model='results/00014-stylegan2-locallyMadeDataset-1gpu-config-f/network-snapshot-010866.pkl'
    _G, _D, Gs = load_stylegan_networks(model)
    image = watchDir('/content/drive/MyDrive/fypImages')
    write_to_file('/content/drive/MyDrive/fypOutput/result.txt', queryDiscriminator(_D,image), image)


if __name__ == "__main__":
    main()