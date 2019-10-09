import keras
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, BatchNormalization
from keras.layers import Dropout, Input, GlobalAveragePooling3D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define additional methods 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def conv3d_attention(input, kernel_size=(3, 3, 7), attention=False):
    conv = Conv3D(filters=8, kernel_size=kernel_size, activation='relu')(input)
    if attention:
        gpool = GlobalAveragePooling2D()(conv[0:2])
        print(gpool.shape)
        #shrink = Dense(size(gpool.s), activation='relu', name='fc' + str(128))(x)

    return conv

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Model 1
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def hsi1(img_shape=(512, 512, 200, 1), pca_input=True, attention=False):

    # Image input
    d0 = Input(shape=img_shape)
    # Convolutional layer 1
    conv_layer1 = conv3d_attention(d0, attention=True)
    #conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(conv_layer1)
    #conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)

hsi1()