import pandas as pd
import numpy as np
import os, sys
import keras
import matplotlib.pyplot as plt
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.models import Model
from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as K

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



#----------------------Parameters----------------------------------------

#dimensions of our images.
img_width, img_height = 224,224 #This matches the MobileNet input size

train_data_dir = '../hotdogs_data/train'
validation_data_dir = '../hotdogs_data/validation'
nb_train_samples = 800 #this divided by the batch size will give the number of steps per epoch
nb_validation_samples = 100
epochs = 50
batch_size = 16 #how many images to include before taking a gradient step

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)



# STEP 1 Build the Model-----------------------------------------------------------------------------------------------------------------


base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more
x=Dropout(0.75)(x)
x=Dense(64,activation='relu')(x)
x=Dropout(0.5)(x)
preds=Dense(1, activation='sigmoid')(x)
#preds=Dense(2,activation='softmax')(x) #final layer with softmax activation


model=Model(inputs=base_model.input,outputs=preds)


#Check model
for i,layer in enumerate(model.layers):
  print(i,layer.name)


#Set the allowed trainable models
for layer in model.layers:
    layer.trainable=False


# or if we want to set the first X layers of the network to be non-trainable
for layer in model.layers[:80]:
    layer.trainable=False
for layer in model.layers[80:]:
    layer.trainable=True

    

#-----------STEP 2 LOAD the training data-------------------------------------



test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')



model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

class_weight={0.0: 1.0, 1.0:1.0}

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    class_weight=class_weight)


model.save('transfer_learning_mobile_net.h5')

#os.system("tensorflowjs_converter --input_format keras transfer_learning_mobile_net.h5 ../model/")
