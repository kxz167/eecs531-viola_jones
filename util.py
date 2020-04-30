import numpy as np
from PIL import Image
import os

def image_from(path):
    if os.path.exists(path):
        return np.array(Image.open(path), dtype=np.int64)
    return None

def images_from_dir(dirpath, limit=float('inf')):
    images = []
    for i, filename in enumerate(os.listdir(dirpath)):
        images.append(image_from(os.path.join(dirpath, filename)))
        if i >= limit - 1:
            break
    return np.asarray(images)

def integral_image(image):
    """
    Computes the integral image representation of a picture. The integral image is defined as following:
    1. s(x, y) = s(x, y-1) + i(x, y), s(x, -1) = 0
    2. ii(x, y) = ii(x-1, y) + s(x, y), ii(-1, y) = 0
    Where s(x, y) is a cumulative row-sum, ii(x, y) is the integral image, and i(x, y) is the original image.
    The integral image is the sum of all pixels above and left of the current pixel
      Args:
        image : an numpy array with shape (m, n)
    """
    ii = np.zeros(image.shape)
    s = np.zeros(image.shape)
    for y in range(len(image)):
        for x in range(len(image[y])):
            s[y][x] = s[y-1][x] + image[y][x] if y-1 >= 0 else image[y][x]
            ii[y][x] = ii[y][x-1]+s[y][x] if x-1 >= 0 else s[y][x]
    return ii

if __name__ == '__main__':
    img = np.ones((5,5))
    print(integral_image(img))
