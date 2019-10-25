import numpy as np
import cv2
import random
import gdal
import matplotlib.pyplot as plt

dim = 256
n_channels = 290


def randomflip(x):
    p = random.randint(0, 1)
    if p:
        x = np.flip(x, 1)
    return x


def randomzoom(x, zoom):
    p = random.randint(0, zoom)
    offset1 = int(x.shape[0] * p / 100.) * 2
    offset2 = int(x.shape[1] * p / 100.) * 2
    x2 = cv2.resize(x[int(offset1 / 2):x.shape[0] - int(offset1 / 2),
                    int(offset2 / 2):x.shape[1] - int(offset2 / 2), :],
                    (dim, dim))
    return x2


addr = 'G:\\MSU\\Intel\\Training_Data\\Avocado_2class\\train\\bvocado_88.tiff'

# Read image
ds = gdal.Open(addr)
rb = ds.GetRasterBand(1)
img_array = rb.ReadAsArray()
img = np.zeros((img_array.shape[0], img_array.shape[1], n_channels))
img[:, :, 0] = img_array
for ind in range(1, n_channels):
    rb = ds.GetRasterBand(ind + 1)
    img_array = rb.ReadAsArray()
    img[:, :, ind] = img_array

imp = np.zeros((img.shape[0], img.shape[1], 3))
imp[:, :, 0] = img[:, :, 121]
imp[:, :, 1] = img[:, :, 77]
imp[:, :, 2] = img[:, :, 33]

plt.figure()
plt.imshow(imp)
axes = plt.gca()
plt.show()


# Applies random transformations
img = randomflip(img)
img2 = randomzoom(img, 20)


imp = np.zeros((img2.shape[0], img2.shape[1], 3))
imp[:, :, 0] = img2[:, :, 121]
imp[:, :, 1] = img2[:, :, 77]
imp[:, :, 2] = img2[:, :, 33]

plt.figure()
plt.imshow(imp)
plt.show()

