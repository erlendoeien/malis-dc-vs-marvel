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
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
   



# Directory with our training dandelion pictures
train_dc_dir = os.path.join('/Users/giannis_pitsiorlas/Documents/Work/EURECOM/MALIS/project/train/DC_proc')

# Directory with our training grass pictures
train_marvel_dir = os.path.join('/Users/giannis_pitsiorlas/Documents/Work/EURECOM/MALIS/project/train/Marvel_proc')

# Directory with our validation dandelion pictures
valid_dc_dir = os.path.join('/Users/giannis_pitsiorlas/Documents/Work/EURECOM/MALIS/project/validation/DC_proc')

# Directory with our validation grass pictures
valid_marvel_dir = os.path.join('/Users/giannis_pitsiorlas/Documents/Work/EURECOM/MALIS/project/validation/Marvel_proc')




train_dc_names = os.listdir(train_dc_dir)
print(train_dc_names[:10])

train_marvel_names = os.listdir(train_marvel_dir)
print(train_marvel_names[:10])

valid_dc_names = os.listdir(valid_dc_dir)
print(valid_dc_names[:10])

valid_marvel_names = os.listdir(valid_marvel_dir)
print(valid_marvel_names[:10])



print('total training dc images:', len(os.listdir(train_dc_dir)))
print('total training marvel images:', len(os.listdir(train_marvel_dir)))
print('total validation dc images:', len(os.listdir(valid_dc_dir)))
print('total validation marvel images:', len(os.listdir(valid_marvel_dir)))





#%%

# Rescaling all imagees by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 10 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/Users/giannis_pitsiorlas/Documents/Work/EURECOM/MALIS/project/train/',  # This is the source directory for training images
        classes = ['DC_proc', 'Marvel_proc'],
        target_size=(200, 200),  # All images will be resized to 200x200
        batch_size=10,
        # Use binary labels
        class_mode='binary')

# Flow validation images in batches of 5 using valid_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        '/Users/giannis_pitsiorlas/Documents/Work/EURECOM/MALIS/project/validation/',  # This is the source directory for training images
        classes = ['DC_proc', 'Marvel_proc'],
        target_size=(200, 200),  # All images will be resized to 200x200
        batch_size=5,
        # Use binary labels
        class_mode='binary',
        shuffle=False)


#%%


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape = (200,200,3)), 
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])

#model.summary()

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_generator,
      steps_per_epoch=10,  
      epochs=50,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)

model.evaluate(validation_generator)

model.save('model.h5')

#%%
# STEP_SIZE_TEST=validation_generator.n//validation_generator.batch_size
# validation_generator.reset()
# preds = model.predict(validation_generator,
#                       verbose=1)
# from sklearn.metrics import roc_auc_score
# from sklearn import svm, datasets
# from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import label_binarize
# from sklearn.multiclass import OneVsRestClassifier
# from scipy import interp
# from sklearn.metrics import roc_auc_score

# fpr, tpr, _ = roc_curve(validation_generator.classes, preds)
# roc_auc = auc(fpr, tpr)






# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#           lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()

#%%

%matplotlib inline

for i in range(1,40):
 
  # predicting images
  path = '/Users/giannis_pitsiorlas/Documents/Work/EURECOM/MALIS/project/test_proc/'
  img = image.load_img(path + str(i) +'.png', target_size=(200, 200))
  x = image.img_to_array(img)
  plt.imshow(x/255.)
  x = np.expand_dims(x, axis=0)
  images = np.vstack([x])
  classes = model.predict(images, batch_size=2)
  print(classes[0])
  if classes[0]>0.5:
    print(str(i) + " is  DC")
  else:
    print(str(i) + " is  Marvel")
    
    
    
    
    






