import numpy as np
import cv2
import random
import gdal
import matplotlib.pyplot as plt

dim = 128
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


def segment_produce(im):

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


def randomcrop(im, d):
    t = 0
    im2 = im
    while t <= 0.8:  # makes sure that more than 80% of the cropped image is different than zero
        x = random.randint(0, im.shape[1] - d)
        y = random.randint(0, im.shape[0] - d)
        im2 = im[y:y+d, x:x+d]
        r, c = np.nonzero(im2[:, :, 0])
        t = c.size / (im2.shape[0] * im2.shape[1])

    # Adds a random rotation
    p = random.randint(0, 1)
    if p:
        im2 = np.rot90(im2)
    return im2


# addr = 'G:\\MSU\\Intel\\Training_Data\\Avocado_2class\\train\\bvocado_95.tiff'
addr = 'G:\\MSU\\Intel\\Training_Data\\Tomato_2class\\fresh\\tomato_4.tiff'

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

num = 260
img2 = cv2.equalizeHist(np.uint8(img[:, :, num] * 255 / (np.max(img[:, :, num]) - np.min(img[:, :, num]))))
img2 = cv2.medianBlur(img2, 3)

plt.figure()
plt.imshow(img2)
plt.show()

msk = segment_produce(img2)

# Calculate xmin, ymin, xmax, ymax
rows, cols = np.nonzero(msk)
img[msk == 0] = np.zeros([1, img.shape[2]])    # change for img
img = img[rows.min():rows.max(), cols.min():cols.max(), :]  # change for img

# Comment from here
# Get color image (R:121, G:77, B:33)
imp = np.zeros((img.shape[0], img.shape[1], 3))
imp[:, :, 2] = img[:, :, 121]
imp[:, :, 1] = img[:, :, 77]
imp[:, :, 0] = img[:, :, 50]
imp = np.uint8(imp * 255 / (np.max(imp) - np.min(imp)))
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
imp[:, :, 0] = cv2.medianBlur(clahe.apply(imp[:, :, 0]), 3)
imp[:, :, 0][imp[:, :, 0] < 15] = 0
imp[:, :, 1] = cv2.medianBlur(clahe.apply(imp[:, :, 1]), 3)
imp[:, :, 1][imp[:, :, 1] < 15] = 0
imp[:, :, 2] = cv2.medianBlur(clahe.apply(imp[:, :, 2]), 3)
imp[:, :, 2][imp[:, :, 2] < 15] = 0
rgb = np.uint8(np.clip(imp, 0, 255))
plt.figure()
plt.imshow(rgb)
plt.show()

cv2.imwrite('G:\\MSU\\Intel\\Images_report\\segment.jpg', rgb)

rgb2 = randomcrop(rgb, dim)
cv2.imwrite('G:\\MSU\\Intel\\Images_report\\random1.jpg', rgb2)
rgb2 = randomcrop(rgb, dim)
cv2.imwrite('G:\\MSU\\Intel\\Images_report\\random2.jpg', rgb2)
rgb2 = randomcrop(rgb, dim)
cv2.imwrite('G:\\MSU\\Intel\\Images_report\\random3.jpg', rgb2)
rgb2 = randomcrop(rgb, dim)
cv2.imwrite('G:\\MSU\\Intel\\Images_report\\random4.jpg', rgb2)
