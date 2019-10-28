
from keras.layers import Conv3D, SeparableConv2D, Dense, Reshape
from keras.layers import Input, GlobalAveragePooling2D, AveragePooling3D, Lambda, Concatenate, Multiply
from keras.models import Model

import keras.backend as k

k.set_image_data_format('channels_last')
k.set_learning_phase(1)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define additional methods 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def conv3d_attention(inp, filters, kernel_size=(3, 3, 7), attention=False):
    conv = Conv3D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(inp)
    if attention:
        branch_outputs = []
        for i in range(conv.shape[-1]):
            # Slicing the ith channel:
            xin = Lambda(lambda x: x[:, :, :, :, i])(conv)
            # Global average of each channel:
            pool = GlobalAveragePooling2D()(xin)
            # Shrink operation
            shrink = Dense(int(pool.shape[-1].value / 4), activation='relu')(pool)
            # Expansion operation
            expansion = Dense(pool.shape[-1].value, activation='sigmoid')(shrink)
            # Scale input
            scaled = Multiply()([xin, expansion])
            scaled = Reshape((scaled.shape[1].value, scaled.shape[2].value, scaled.shape[3].value, 1))(scaled)
            branch_outputs.append(scaled)
        # Concatenating together the per-channel results:
        conv = Concatenate()(branch_outputs)

    return conv


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Model 1
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def hsi1(img_shape=(256, 256, 50, 1), attention=True):
    # Input
    d0 = Input(shape=img_shape)

    # 3D convolutions
    conv_layer1 = conv3d_attention(d0, 8, attention=attention)
    conv_layer2 = conv3d_attention(conv_layer1, 8, attention=attention)
    conv_in = Concatenate()([conv_layer1, conv_layer2])
    conv_layer3 = conv3d_attention(conv_in, 8, attention=attention)
    conv_in = Concatenate()([conv_in, conv_layer3])
    conv_layer4 = conv3d_attention(conv_in, 8, attention=attention)
    conv_in = Concatenate()([conv_in, conv_layer4])
    conv_in = AveragePooling3D(pool_size=(3, 3, 8), strides=2)(conv_in)

    # 2D convolutions
    conv_in = Reshape((conv_in.shape[1].value, conv_in.shape[2].value,
                       conv_in.shape[3].value * conv_in.shape[4].value))(conv_in)
    conv_in = SeparableConv2D(128, kernel_size=3, strides=(2, 2), padding='same',
                              dilation_rate=3)(conv_in)
    conv_in = SeparableConv2D(128, kernel_size=3, strides=(2, 2), padding='same',
                              dilation_rate=3)(conv_in)
    conv_in = SeparableConv2D(128, kernel_size=3, strides=(2, 2), padding='same',
                              dilation_rate=3)(conv_in)
    conv_in = SeparableConv2D(128, kernel_size=3, strides=(2, 2), padding='same',
                              dilation_rate=3)(conv_in)
    print(conv_in.shape)
    pool = GlobalAveragePooling2D()(conv_in)
    fc1 = Dense(32, activation='relu')(pool)
    fc1 = Dense(1, activation='sigmoid')(fc1)

    return Model(d0, fc1)


net = hsi1(img_shape=(256, 256, 10, 1))
net.summary()
