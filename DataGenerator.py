# -*- coding: utf-8 -*-
"""
Keras Data Generator for Data Augmentation
"""
import numpy as np
import keras
import cv2
import random
import gdal


class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, batch_size=1, dim=128, n_channels=200,
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

        p = random.randint(0, 1)
        if p:
            x = np.flip(x, 0)
        return x

    def randomzoom(self, x, zoom, dim):
        p = random.randint(0, zoom)
        offset1 = int(x.shape[0] * p / 100.) * 2
        offset2 = int(x.shape[1] * p / 100.) * 2
        x2 = cv2.resize(x[int(offset1 / 2):x.shape[0] - int(offset1 / 2),
                        int(offset2 / 2):x.shape[1] - int(offset2 / 2), :],
                        (dim, dim))
        return x2

    def segment_produce(self, im):

        # Apply Grab-Cut
        y = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        # Rectange values: start x, start y, width, height
        rectangle = (10, 10, y.shape[1] - 20, y.shape[0] - 20)
        # Create initial mask
        mask = np.zeros(y.shape[:2], np.uint8)

        # Create temporary arrays used by grabCut
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Run grabCut
        cv2.grabCut(y,  # Our image
                    mask,  # The Mask
                    rectangle,  # Our rectangle
                    bgdModel,  # Temporary array for background
                    fgdModel,  # Temporary array for background
                    5,  # Number of iterations
                    cv2.GC_INIT_WITH_RECT)  # Initiative using our rectangle

        # Create mask where sure and likely backgrounds set to 0, otherwise 1
        mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Multiply image with new mask to subtract background
        image_rgb_nobg = y * mask_2[:, :, np.newaxis]

        # Morphological erosion and dilation
        kernel = np.ones((7, 7), np.uint8)
        image_rgb_nobg = cv2.erode(image_rgb_nobg, kernel, iterations=1)
        image_rgb_nobg = cv2.dilate(image_rgb_nobg, kernel, iterations=1)

        mask = mask * 0
        mask[(image_rgb_nobg[:, :, 0] == 0) & (image_rgb_nobg[:, :, 1] == 0) & (image_rgb_nobg[:, :, 2] == 0)] = 255
        mask = 255 - mask

        return mask

    def randomcrop(self, im, d):
        t = 0
        im2 = im
        while t <= 0.8:  # makes sure that more than 80% of the cropped image is different than zero
            x = random.randint(0, im.shape[1] - d)
            y = random.randint(0, im.shape[0] - d)
            im2 = im[y:y + d, x:x + d]
            r, c = np.nonzero(im2[:, :, 0])
            t = c.size / (im2.shape[0] * im2.shape[1])

        # Adds a random rotation
        p = random.randint(0, 1)
        if p:
            im2 = np.rot90(im2)
        return im2

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

            # Applies random flip (horizontal and vertical)
            img = self.randomflip(img)

            # Extract that band 260 to segment the produce and remove background
            num = 260
            band2segment = cv2.equalizeHist(np.uint8(img[:, :, num] * 255 /
                                                     (np.max(img[:, :, num]) - np.min(img[:, :, num]))))
            band2segment = cv2.medianBlur(band2segment, 3)  # Applies median filter to reduce noise
            msk = self.segment_produce(band2segment)

            # Calculate bounding box in real image
            rows, cols = np.nonzero(msk)
            img[msk == 0] = np.zeros([1, img.shape[2]])
            img = img[rows.min():rows.max(), cols.min():cols.max(), :]

            # Apply random crop
            img = self.randomcrop(img, self.dim)

            x[i, ] = np.reshape(img, (self.dim, self.dim, self.n_channels, 1))
            y[i, ] = [0 if 'b' in addr else 1]

        return x, y
