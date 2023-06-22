import tensorflow as tf
import os

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

import cv2 #open cv
import imghdr #import hdr

data_dir = 'data'
image_exts= ['jpeg', 'jpg', 'bpm', 'png']

for image_class in os.listdir(data_dir):
    #skip the .DS_Store file
    if image_class == '.DS_Store':
        continue
    #loop through every image in the folder
    filepath = os.path.join(data_dir,image_class)
    if os.path.isfile(filepath):
        for image in os.listdir(filepath):
            image_path = os.path.join(data_dir, image_class, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    print("Image not in ext list {}".format(image_path))
                    #remove the image if it does not exist
                    os.remove(image_path)
            except Exception as e:
                print('Issue with image {}'.format(image_path))
                #os.remove(image_path)
  
import numpy as np
from matplotlib import pyplot as plt

data = tf.keras.utils.image_dataset_from_directory('data')
data_iterator = data.as_numpy_iterator() ### getting an error
batch = next(data_iterator) #Unknown image file format. One of JPEG, PNG, GIF, BMP required.


