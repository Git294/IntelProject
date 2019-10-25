import glob
import os
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
# from keras.utils import multi_gpu_model

from DataGenerator import DataGenerator
from networks import hsi1

import keras.backend as k
k.set_image_data_format('channels_last')
k.set_learning_phase(1)

# import pickle

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Lists
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Obtiene dirección del proyecto y de la carpeta que contiene al proyecto
basepath = os.getcwd()[:-13]

# Obtiene listas de imágenes de entrenamiento, valdiación y test
orig_path_train = basepath + '\\Training_Data\\Avocado_2class\\train\\*.tiff'
orig_path_val = basepath + '\\Training_Data\\Avocado_2class\\val\\*.tiff'

# Obtiene una lista de las direcciones de las imágenes y sus máscaras
addri_train = sorted(glob.glob(orig_path_train))
addri_val = sorted(glob.glob(orig_path_val))

# Parametros para la generación de data
n_channels = 290
dim = 256
params = {'dim': dim,
          'batch_size': 1,
          'n_channels': n_channels,
          'shuffle': True}

# Crea diccionarios
data_dict = {"train": addri_train, "validation": addri_val}

# Generadores
training_generator = DataGenerator(data_dict['train'], **params)
validation_generator = DataGenerator(data_dict['validation'], **params)

# Guarda generadores
# with open('Generators/Train', 'wb') as f:
#     pickle.dump(training_generator, f)
#
# with open('Generators/Validation', 'wb') as f:
#     pickle.dump(validation_generator, f)


# Si los generadores ya han sido creados con anterioridad, sólo se cargan
# with open('Generators/Train', 'rb') as f:
#     training_generator = pickle.load(f)
# with open('Generators/Validation', 'rb') as f:
#     validation_generator = pickle.load(f)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Entrenamiento
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Load model
print("Cargando modelo...")
model = hsi1(img_shape=(dim, dim, n_channels, 1))
model.summary()

# Compile model
optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

# checkpoint
filepath = "weights-hs1-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Train model on dataset
print("Empieza entrenamiento...")
history = model.fit_generator(generator=training_generator,
                              validation_data=validation_generator,
                              use_multiprocessing=False,
                              workers=1,
                              shuffle=True,
                              epochs=150,
                              max_queue_size=2,
                              callbacks=callbacks_list)

print("Terminó entrenamiento!!!!")
