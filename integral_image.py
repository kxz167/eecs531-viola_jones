#For image processing
from PIL import Image
#For command line arguments
import sys, getopt
#For numpy arrays:
import numpy as np
from numpy import asarray
#For surfaceplot:
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    main(sys.argv[1])