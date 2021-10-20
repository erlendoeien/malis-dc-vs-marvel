#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: giannis_pitsiorlas
"""

import os
import pandas as pd
import numpy as np 
import itertools
import keras
import tensorflow as tf
from keras.preprocessing import image
from tensorflow import keras
from keras.models import load_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from keras.models import Sequential 
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense 
from keras import applications 
from keras.utils.np_utils import to_categorical 
from tensorflow.keras.optimizers import RMSprop
from keras.models import Sequential
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import math 
import datetime
import time
from sklearn.metrics import classification_report 


base_dir = '/Users/giannis_pitsiorlas/Documents/Work/EURECOM/MALIS'
#creating the paths and directories 
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

#directory with the training waters and lands pictures
train_dc_dir = os.path.join(train_dir, 'DC')
train_marvel_dir = os.path.join(train_dir, 'Marvel')

#directory with the validation waters and lands pictures
validation_waters_dir = os.path.join(validation_dir,'DC')                                     
validation_lands_dir = os.path.join(validation_dir,'Marvel')

#all images rescalation by 1/255
train_datagen = ImageDataGenerator(rescale = 1.0/255. )
test_datagen = ImageDataGenerator(rescale = 1.0/255. ) 


#flow training images in batches of 2
train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 2, 
                                                    class_mode = 'binary', 
                                                    target_size =(60, 60))
#flow validation images in batches of 2
validation_generator = train_datagen.flow_from_directory(validation_dir, batch_size = 2, 
                                                    class_mode = 'binary', 
                                                    target_size =(60, 60))

#construction of the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(60,60,3)),
    #tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

#training the model

model.compile(optimizer = RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['acc'])


history = model.fit_generator(train_generator,
                              validation_data = validation_generator,
                              steps_per_epoch = 10,
                              epochs = 7,
                              validation_steps = 2,
                              verbose = 1)




model.save('model.h5')

