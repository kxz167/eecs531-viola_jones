#For image processing
from PIL import Image
#For command line arguments
import sys, getopt
#For numpy arrays:
import numpy as np
from numpy import asarray
#For surfaceplot:
import matplotlib.pyplot as plt
from typing import Tuple, List

def main(argname):
    image = Image.open(argname).convert('L')
    image_array = asarray(image)
    
    integral_array = calc_int(image_array)
    
    plt.imshow(integral_array)
    plt.show()

def calc_int(img_array):
    integral_array = np.zeros_like(img_array, dtype="long")

    for (x,y), val in np.ndenumerate(img_array):

        above_val = 0
        left_val = 0
        corner_val = 0
        
        #Border case handling:
        if y-1 >= 0:
            left_val = integral_array[x,y-1]
        if x-1 >= 0:
            above_val = integral_array[x-1,y]
        if x-1 >= 0 and y-1 >= 0:
            corner_val = integral_array[x-1, y-1]

        integral_array[x,y] = val + above_val + left_val - corner_val
    return integral_array

def region_sum(integral_image, region: Tuple[Tuple[int]]):
    '''
    Calculate points between x0, y0 and x1, y1 (not including the x1, y1 row and col)
    '''
    assert len(region) == 2, 'Assuming top left (A), bottom right(D) points for region'
    A = region[0]
    D = region[1]
    C = (A[0], D[1])
    B = (D[0], A[1])
    # Require some padding to due to indexing problem
    copy = np.copy(integral_image)
    copy = np.pad(copy, ((1, 0), (1, 0)))
    # Following ii(D) + ii(A) - (ii(C) + ii(B)) of viola jones
    return copy[D] + copy[A] - (copy[B] + copy[C])

# def to_integral_image(img_arr):
#     """
#     Calculates the integral image based on this instance's original image data.
#     :param img_arr: Image source data
#     :type img_arr: numpy.ndarray
#     :return Integral image for given image
#     :rtype: numpy.ndarray
#     """
#     # an index of -1 refers to the last row/column
#     # since row_sum is calculated starting from (0,0),
#     # rowSum(x, -1) == 0 holds for all x
#     row_sum = np.zeros(img_arr.shape)
#     # we need an additional column and row
#     integral_image_arr = np.zeros((img_arr.shape[0] + 1, img_arr.shape[1] + 1))
#     for x in range(img_arr.shape[1]):
#         for y in range(img_arr.shape[0]):
#             row_sum[y, x] = row_sum[y-1, x] + img_arr[y, x]
#             integral_image_arr[y+1, x+1] = integral_image_arr[y+1, x-1+1] + row_sum[y, x]
#     return integral_image_arr

if __name__ == "__main__":
    # Test code
    img = np.ones((3, 3))
    ii = calc_int(img)
    sum_reg = region_sum(ii, [(0, 0), (1, 1)])
    assert sum_reg == ii[0, 0], 'sum is {}'.format(sum_reg)
    sum_reg = region_sum(ii, [(0, 0), (-1, -1)])
    assert sum_reg == np.sum(img), 'sum is {}'.format(sum_reg)