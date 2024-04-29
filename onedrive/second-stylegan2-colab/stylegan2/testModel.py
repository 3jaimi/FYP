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


def find_pkl_files(mother_dir):
    
    pkl_files = []
    for subdir, dirs, files in os.walk(mother_dir):
        for file in files:
            if file.endswith(".pkl") and file not in doneFiles:
                pkl_files.append(os.path.join(subdir, file))
    return pkl_files

def find_pkl_files_in_dir(mother_dir):
    pkl_files = []
    for file in os.listdir(mother_dir):
        if file.endswith(".pkl"):
            pkl_files.append(os.path.join(mother_dir, file))
    return pkl_files

doneFiles=['network-snapshot-10000.pkl','network-snapshot-010000.pkl','network-snapshot-010016.pkl','network-snapshot-010032.pkl','network-snapshot-010049.pkl','network-snapshot-010065.pkl','network-snapshot-010082.pkl','network-snapshot-010098.pkl','network-snapshot-010114.pkl','network-snapshot-010131.pkl','submit_config.pkl', 'network-snapshot-010147.pkl','network-snapshot-010163.pkl','network-snapshot-010195.pkl','network-snapshot-010179.pkl','network-snapshot-010244.pkl','network-snapshot-010228.pkl','network-snapshot-010211.pkl','network-snapshot-010866.pkl','network-snapshot-010849.pkl','network-snapshot-010833.pkl']


def load_stylegan_networks(pkl_file):
    tflib.init_tf()
    print('Loading networks from "%s"...' % pkl_file)
    with open(pkl_file, 'rb') as f:
        _G, _D, Gs = pickle.load(f)

    return _G, _D, Gs

def create_dataset(path):
    datagen = ImageDataGenerator(rescale=1./255)
    image_generator = datagen.flow_from_directory(
            path,
            target_size=(1024, 1024),
            batch_size=1,
            class_mode='binary',
            shuffle=False)  # _Disable shuffling to keep images grouped by class

    images = []
    labels = []
    for _ in range(len(image_generator)):
        batch_images, batch_labels = next(image_generator)
        images.extend(batch_images)
        labels.extend(batch_labels)

    # Convert lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    # _Group images by their labels into two arrays
    class_0_images = images[labels == 0]
    class_1_images = images[labels == 1]

    # Transpose images
    class_0_images = tf.transpose(class_0_images, perm=[0, 3, 2, 1])
    class_1_images = tf.transpose(class_1_images, perm[0, 3, 2, 1])

    # Convert images to tensors
    #class_0_images = tf.convert_to_tensor(class_0_images, dtype=tf.float32)
    #class_1_images = tf.convert_to_tensor(class_1_images, dtype=tf.float32)

    return class_0_images, class_1_images

def queryDiscriminator(model, dir):
    image_paths=os.listdir(dir)
    scores=[]
    for imagePath in image_paths:
        image = Image.open(dir+imagePath)
        image = np.transpose(np.asarray(image), (2,0,1))[None]
        labels = np.array([]).reshape(1,0)
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels)
        images = misc.adjust_dynamic_range(image, [0, 255], [-1, 1])
        score = model.get_output_for(images, labels,is_training=False)
        #probability = tf.math.softplus(score)
        parsedScore=score.eval()[0][0]
        print(parsedScore)
        scores.append(parsedScore)
    numpScores=np.array(scores)
    print("MEAN SCORE: ", numpScores.mean())




def printTensor(tensor):
        for i in range(tensor.shape[0]):
            print("Entry {}: {}".format(i, tensor[i]))

def draftFunc(model,images):
    for image in images:
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        labels = np.array([]).reshape(1,0)
        labels = tf.convert_to_tensor(labels)
        scores_out = model.get_output_for(image, labels,is_training=False)
        probability = tf.math.softplus(scores_out)
        prob=probability.eval()
        print("probability:{} ({})".format(probability,scores_out))
    #scores_out = autosummary('Loss/scores', scores_out)
    #fake_scores_out = model.get_output_for(fake_images_out, labels, is_training=True)
    return scores_out


def print_network_summaries(_G, _D, _Gs):
    #print("_Generator Summary:")
    #_G.print_layers()

    #print("\n_Generator Synthesis Network Summary:")
    #_Gs.print_layers()

    print("\n_Discriminator Summary:")
    _D.print_layers()


def main():

    GANList=find_pkl_files_in_dir('results/00015-stylegan2-locallyMadeDataset-1gpu-config-f/')
    for model in GANList:
        _G, _D, Gs = load_stylegan_networks(model)
        print("Networks loaded successfully!")
        print('\nREALS:\n')
        queryDiscriminator(_D,'testDatasets/reals/')
        print('\nMODEL-FAKES:\n')
        queryDiscriminator(_D,'testDatasets/modelFakes/')
        print('\nFF++-FAKES:\n')
        queryDiscriminator(_D,'testDatasets/FF++Fakes/')
    

    #print("Printing network summaries...")
    #print_network_summaries(_G, _D, Gs)

if __name__ == "__main__":
    main()