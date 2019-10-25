"""
https://becominghuman.ai/not-hotdog-with-keras-and-tensorflow-js-fab138fe7e84

epochs how many times we iterate through the "whole" data set

"""


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True






#dimensions of our images.
img_width, img_height = 150,150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 600 #this divided by the batch size will give the number of steps per epoch
nb_validation_samples = 100
epochs = 200
batch_size = 120 #how many images to include before taking a gradient step

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


#Network Design
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rotation_range=40,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
    fill_mode='nearest',
    rescale=1. / 255
)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)


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


#adjust the class weights to match the training samples
#0 is hotdog, 1 is not hotdog

#This makes the weight of assigning a hotdog 10 times higher as we have less training data
#for hotdogs

class_weight={0.0: 1.0, 1.0:1.0}



model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    class_weight=class_weight)


model.save('balanced_training.h5')

#os.system("tensorflowjs_converter --input_format keras first_try.h5 model/")

#tfjs.converters.save_keras_model(model, "./model/")
