from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import load_model
from keras.applications.mobilenet import preprocess_input

import cv2
import numpy as np

import os
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from os import listdir
from os.path import isfile, join


model = load_model('transfer_learning_mobile_net.h5')


hotdogs=0
total=0

def predict_filename(fn):

    
    img = cv2.imread(fn)
    img = cv2.resize(img,(150,150))
    img = np.reshape(img,[1,150,150,3])
    img=preprocess_input(img)
        
    classes = model.predict(img)[0]
    #print("Class: ", classes[0])
    
    
    if classes<=0.50:
    
        return(classes[0], "Hotdog")
    else:
        return(classes[0], "Not Hotdog")




onlyfiles = [f for f in listdir(sys.argv[1]) if isfile(join(sys.argv[1], f))]

for f in onlyfiles:
    fn=sys.argv[1]+f
    p, ans=predict_filename(fn)
    total+=1
    if (ans=="Hotdog"):
        print(f, ans, p)
        hotdogs+=1
    else:
        print(f, ans,p)
        

percent_hd=(float(hotdogs))/total*100.0

print("Percent Hotdogs in this directory:", percent_hd)
