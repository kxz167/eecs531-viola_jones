import numpy as np
from PIL import Image
import os

def image_from(path):
    if os.path.exists(path):
        return np.array(Image.open(path))
    return None