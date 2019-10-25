# -*- coding: utf-8 -*-
"""
GENERADOR DE DATA EN KERAS
"""
import numpy as np
import keras
import cv2
import random
import gdal


class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, batch_size=1, dim=256, n_channels=200,
                 shuffle=True):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Calculate the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate a batch"""
        # Genera los índices del batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Crea una lista de IDs correspondientes a indexes
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Genera la data
        x, y = self.__data_generation(list_IDs_temp)

        return x, y

    def on_epoch_end(self):
        """Update indexes"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def randomflip(self, x):
        p = random.randint(0, 1)
        if p:
            x = np.flip(x, 1)
        return x

    def randomzoom(self, x, zoom):
        p = random.randint(0, zoom)
        offset1 = int(x.shape[0] * p / 100.) * 2
        offset2 = int(x.shape[1] * p / 100.) * 2
        x2 = cv2.resize(x[int(offset1 / 2):x.shape[0] - int(offset1 / 2),
                        int(offset2 / 2):x.shape[1] - int(offset2 / 2), :],
                        (self.dim, self.dim))
        return x2

    def __data_generation(self, list_IDs_temp):
        """Genera data"""  # X : (n_samples, *dim, n_channels)
        # Initialize input
        x = np.empty((self.batch_size, self.dim, self.dim, self.n_channels, 1), dtype=np.float)
        y = np.empty((self.batch_size, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            addr = ID
            # Read image
            ds = gdal.Open(addr)
            rb = ds.GetRasterBand(1)
            img_array = rb.ReadAsArray()
            img = np.zeros((img_array.shape[0], img_array.shape[1], self.n_channels))
            img[:, :, 0] = img_array
            for ind in range(1, self.n_channels):
                rb = ds.GetRasterBand(ind + 1)
                img_array = rb.ReadAsArray()
                img[:, :, ind] = img_array

            # Applies random transformations
            img = self.randomflip(img)
            img = self.randomzoom(img, 10)

            # basepath = os.getcwd()[:-7]
            # cv2.imwrite(basepath + '//Pruebas//' + os.path.basename(addr)[:-4] + "_orig.png", img2)
            # cv2.imwrite(basepath + '//Pruebas//' + os.path.basename(addr)[:-4] + "_mask.png", mask2*255)

            x[i, ] = np.reshape(img, (self.dim, self.dim, self.n_channels, 1))
            y[i, ] = [0 if 'b' in addr else 1]

        return x, y
