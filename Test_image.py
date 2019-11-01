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

    p = random.randint(0, 1)
    if p:
        x = np.flip(x, 0)
    return x


def randomzoom(x, zoom):
    p = random.randint(0, zoom)
    offset1 = int(x.shape[0] * p / 100.) * 2
    offset2 = int(x.shape[1] * p / 100.) * 2
    x2 = cv2.resize(x[int(offset1 / 2):x.shape[0] - int(offset1 / 2),
                    int(offset2 / 2):x.shape[1] - int(offset2 / 2), :],
                    (dim, dim))
    return x2


addr = 'G:\\MSU\\Intel\\Training_Data\\Avocado_2class\\train\\avocado_2.tiff'

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

# If height > width, rotate
if img.shape[0] > img.shape[1]:
    img = np.rot90(img)


# Applies random transformations
img = randomflip(img)
# img2 = randomzoom(img, 10)

# Printo color image (R:121, G:77, B:33)
imp = np.zeros((img.shape[0], img.shape[1], 3))
imp[:, :, 0] = img[:, :, 121]
imp[:, :, 1] = img[:, :, 77]
imp[:, :, 2] = img[:, :, 50]
imp = np.uint8(imp * 255 / (np.max(imp) - np.min(imp)))
rgb = np.uint8(np.clip(imp*2, 0, 255))
plt.figure()
plt.imshow(rgb)
plt.show()

# Mean image and region growing
#mean_image = np.mean(img, axis=2)
#mean_image = np.uint8(mean_image * 255 / (np.max(mean_image) - np.min(mean_image)))
#plt.figure()
#plt.imshow(mean_image)
#plt.show()

# Basic threshold
# Convert from rgb to grayscale
#gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
rgb2 = (rgb[:, :, 0].astype(np.float) + rgb[:, :, 1].astype(np.float)).astype(np.uint8)
blur = cv2.GaussianBlur(rgb2, (3, 3), 0)
ret3, th3 = cv2.threshold(rgb2, 15, 255, cv2.THRESH_BINARY)
kernel = np.ones((3, 3), np.uint8)
th3 = cv2.dilate(th3, kernel, iterations=1)
kernel = np.ones((8, 8), np.uint8)
th3 = cv2.erode(th3, kernel, iterations=1)
plt.figure()
plt.imshow(th3)
plt.show()

# Deletes the line at the bottom
if th3[0, 0] == 255 or th3[0, -1] == 255:
    count = 1
    while th3[count, 0] == 255 or th3[count, -1]:
        count += 1
    th3[0:count+10, :] = 0
elif th3[-1, 0] == 255 or th3[-1, -1]:
    count = th3.shape[0] - 1
    while th3[count, 0] == 255 or th3[count, -1]:
        count -= 1
    th3[count-10:, :] = 0
plt.figure()
plt.imshow(th3)
plt.show()

# Calculate xmin, ymin, xmax, ymax
rows, cols = np.nonzero(th3)

y = imp
# Rectange values: start x, start y, width, height
rectangle = (cols.min() - 5, rows.min() - 5, cols.max() - cols.min() + 5, rows.max() - rows.min() + 5)
#rectangle = (rows.min(), cols.min(), rows.max() - rows.min(), cols.max() - cols.min())
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

plt.figure()
plt.imshow(image_rgb_nobg)
plt.show()
