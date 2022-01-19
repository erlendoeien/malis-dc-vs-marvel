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
train_dc_dir = os.path.join('/Users/giannis_pitsiorlas/Documents/Work/EURECOM/MALIS/project/6_999x999/train/dc')

# Directory with our training grass pictures
train_marvel_dir = os.path.join('/Users/giannis_pitsiorlas/Documents/Work/EURECOM/MALIS/project/6_999x999/train/marvel')

# Directory with our validation dandelion pictures
valid_dc_dir = os.path.join('/Users/giannis_pitsiorlas/Documents/Work/EURECOM/MALIS/project/6_999x999/validation/dc')

# Directory with our validation grass pictures
valid_marvel_dir = os.path.join('/Users/giannis_pitsiorlas/Documents/Work/EURECOM/MALIS/project/6_999x999/validation/marvel')




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
#searching for any non-image files
import os
from PIL import Image
folder_path = '/Users/giannis_pitsiorlas/Documents/Work/EURECOM/MALIS/project/6_999x999/train'
extensions = []
for fldr in os.listdir(folder_path):
    sub_folder_path = os.path.join(folder_path, fldr)
    for filee in os.listdir(sub_folder_path):
        file_path = os.path.join(sub_folder_path, filee)
        print('** Path: {}  **'.format(file_path), end="\r", flush=True)
        im = Image.open(file_path)
        rgb_im = im.convert('RGB')
        if filee.split('.')[1] not in extensions:
            extensions.append(filee.split('.')[1])




#%%

# Rescaling all imagees by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)
testing_datagen = ImageDataGenerator(rescale=1/255)

train_datagen = ImageDataGenerator(rotation_range=90, fill_mode='nearest')
validation_datagen = ImageDataGenerator(rotation_range=90, fill_mode='nearest')
testing_datagen = ImageDataGenerator(rotation_range=90, fill_mode='nearest')

# Flow training images in batches of 10 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/Users/giannis_pitsiorlas/Desktop/crop_6_400x400_Laplacian_filter/train/',  # This is the source directory for training images
        classes = ['dc', 'marvel'],
        target_size=(256, 256),  # All images will be resized to 200x200
        batch_size=64,
     #   color_mode='grayscale',
        # Use binary labels
        class_mode='binary')

# Flow validation images in batches of 5 using valid_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        '/Users/giannis_pitsiorlas/Desktop/crop_6_400x400_Laplacian_filter/validation/',  # This is the source directory for training images
        classes = ['dc', 'marvel'],
        target_size=(256, 256),  # All images will be resized to 200x200
        batch_size=64,
    #    color_mode='grayscale',
        # Use binary labels
        class_mode='binary',
        shuffle=False)


#%%



model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu')) #128
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu')) #512
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

#model.summary()

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_generator,
      steps_per_epoch=50,  
      epochs=10,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)

model.evaluate(validation_generator)

model.save('rgb_model.h5')


#%%
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig("train_val_acc_rgb.pdf")


plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("train_val_loss_rgbpdf")

plt.show()


#%%
#testing generator 
testing_datagen = ImageDataGenerator(rotation_range=90, fill_mode='nearest')


testing_generator = testing_datagen.flow_from_directory(
        '/Users/giannis_pitsiorlas/Documents/Work/EURECOM/MALIS/project/6_999x999/test/',  # This is the source directory for training images
        classes = ['dc', 'marvel'],
        target_size=(256, 256),  # All images will be resized to 200x200
        batch_size=64,
       # color_mode='grayscale',
        # Use binary labels
        class_mode='binary',
        shuffle=False)

res = model.evaluate(testing_generator)




#%%

#for manual testing
%matplotlib inline
model = keras.models.load_model('/Users/giannis_pitsiorlas/Documents/Work/EURECOM/MALIS/project/rgb_model.h5')

for i in range(1,12):
 
  # predicting images
  path = '/Users/giannis_pitsiorlas/Desktop/Majority Voting test/set 2/marvel/'
  img = image.load_img(path + str(i) +'.png', target_size=(256, 256))
  x = image.img_to_array(img)
  plt.imshow(x/255.)
  x = np.expand_dims(x, axis=0)
  images = np.vstack([x])
  classes = model.predict(images, batch_size=2)
  print(classes[0])
  if classes[0]<0.5:
    print(str(i) + " is  DC")
  else:
    print(str(i) + " is  Marvel")
    
    
#%%



    
    
    






