#%%
import numpy as np 
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from keras.models import Sequential 
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt 
from pathlib import Path
   
#%%
data_dir = Path("data/sets/6_999x999/")
data_gen_kwargs = {"rescale":1/255, "rotation_range":90, "fill_mode": 'nearest'}

train_datagen = ImageDataGenerator(**data_gen_kwargs)
validation_datagen = ImageDataGenerator(**data_gen_kwargs)
test_datagen = ImageDataGenerator(**data_gen_kwargs)

img_width, img_height = 256, 256
batch_size = 64

#%%
# Class names are inferred
dataset_kwargs = {"class_mode": "binary", "target_size": (img_width, img_height), "color_mode": "rgb", "batch_size": batch_size, "seed":1337}

train_generator = train_datagen.flow_from_directory(
        data_dir / "train",
        **dataset_kwargs,)

validation_generator = validation_datagen.flow_from_directory(
        data_dir / "validation",
        **dataset_kwargs)

test_generator = validation_datagen.flow_from_directory(
        data_dir / "test",
        **dataset_kwargs)






#%%

model = Sequential([
  Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
  MaxPooling2D((2, 2)),
  Dropout(0.3),
  Conv2D(64, (3, 3), activation='relu'),
  MaxPooling2D((2, 2)),
  Conv2D(64, (3, 3), activation='relu'), #128
  MaxPooling2D((2, 2)),
  Flatten(),
  Dense(256, activation='relu'), #512
  Dense(1, activation='sigmoid'),
])

model.summary()

#%%
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])


#%%
history = model.fit(train_generator,
      steps_per_epoch=50,  
      epochs=10,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)

#%%
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
# Test the model
res = model.evaluate(test_generator)




#%%

# for manual Majority vote testing 
%matplotlib inline
model = tf.keras.models.load_model('/Users/giannis_pitsiorlas/Documents/Work/EURECOM/MALIS/project/rgb_model.h5')

for i in range(1,12):
  # predicting images
  majority_vote_path = Path('/Users/giannis_pitsiorlas/Desktop/Majority Voting test/set 2/marvel/')
  img = load_img(str(majority_vote_path / str(i) +'.png'), target_size=(256, 256))
  x = img_to_array(img)
  plt.imshow(x/255.)
  x = np.expand_dims(x, axis=0)
  images = np.vstack([x])
  classes = model.predict(images, batch_size=2)
  print(classes[0])
  if classes[0]<0.5:
    print(str(i) + " is  DC")
  else:
    print(str(i) + " is  Marvel")
    
    


    
    
    






