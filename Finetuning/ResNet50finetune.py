
# coding: utf-8

# In[1]:

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
#from keras.applications.resnet50 import ResNet50
from myresnet50 import ResNet50

def ResNet50finetune():
    #Import model without top
    model = ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(224,224,3)))

    #Top of the model
    x = model.output
    x = Flatten(name='flatten')(x) 
    x = BatchNormalization(axis=1, name='batch_norm')(x)
    x = Dense(344, name='fc1000',init='lecun_uniform')(x)
    x = Activation("softmax",name='bfc1000')(x)

    #Final model
    final_model = Model(input=model.input, output=x)

    # to non-trainable (weights will not be updated)
    for layer in final_model.layers[:175]:
        layer.trainable = False
        #Does not help in setting mode to 1 for B.N. Layer  
	#if "BatchNormalization" in str(layer):
	#	layer.mode = 1
    
    return final_model

