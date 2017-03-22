
# coding: utf-8

# In[ ]:

import numpy as np

from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization, Activation
from keras.preprocessing import image
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.regularizers import l2
from keras.models import Sequential
from keras.applications.vgg16 import VGG16

def VGG16finetune():
    #Import model without top
    model = VGG16(include_top=False, weights='imagenet', input_tensor=Input(shape=(224,224,3)))

    #Top of the model
    x = model.output
    x = Flatten(name='flatten')(x) 
    x = BatchNormalization(axis=1, name='batch_norm')(x)
    x = Dense(344, name='fc1000',init='lecun_uniform')(x)
    x = Activation("softmax",name='bfc1000')(x)

    #Final model
    final_model = Model(input=model.input, output=x)

    # to non-trainable (weights will not be updated)
    for layer in final_model.layers[:19]:
        layer.trainable = False
    
    return final_model


