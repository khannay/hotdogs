from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import load_model

import cv2
import numpy as np

import os
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from os import listdir
from os.path import isfile, join


model = load_model('balanced_training.h5')


def predict_filename(fn):

    try:
        img = cv2.imread(fn)
        img = cv2.resize(img,(150,150))
        img = np.reshape(img,[1,150,150,3])

        classes = model.predict_classes(img)[0]

        if classes==0:
            return("Hotdog")
        else:
            return("Not Hotdog")

    except:
        print("Problem with img", fn)

onlyfiles = [f for f in listdir(sys.argv[1]) if isfile(join(sys.argv[1], f))]

for f in onlyfiles:
    fn=sys.argv[1]+f
    print(predict_filename(fn))
    
