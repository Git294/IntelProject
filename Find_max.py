import glob
import os
import numpy as np
from DataGenerator import DataGenerator

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
dim = 128
batch_size = 1
params = {'dim': dim,
          'batch_size': batch_size,
          'n_channels': n_channels,
          'shuffle': True}

# Crea diccionarios
data_dict = {"train": addri_train, "validation": addri_val}

# Generators
training_generator = DataGenerator(data_dict['train'], **params) # comment lines 161-164 of DataGenerator.py
validation_generator = DataGenerator(data_dict['validation'], **params)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Run generators
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

maxs = np.zeros((290, 1))
count = 0
for X_batch, y_batch in training_generator:

    count += 1
    print("Analizing image #" + str(count) + "/" + str(len(addri_train)))

    for i in range(0, 289):
        temp = np.max(X_batch[:, :, :, i, :])
        if temp > maxs[i]:
            maxs[i] = temp

print("Maximum values found")
