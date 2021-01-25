import numpy as np
from classifiers import *
from pipeline import *

from keras.preprocessing.image import ImageDataGenerator

# 1 - Load the model and its pretrained weights
classifier = Meso4()

# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

dataGenerator = ImageDataGenerator(rescale=1. / 255)
import os
print(os.getcwd())
generator = dataGenerator.flow_from_directory(
    'deepfake_database\\train_test',
    target_size=(256, 256),
    batch_size=1,
    class_mode='binary',
    subset='training')

history = classifier.model.fit(generator, epochs=1)
import time
classifier.model.save("{}_{}.h5".format(classifier.__class__.__name__, time.time()))
