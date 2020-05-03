import numpy as np
from PIL import Image
import os
import json
import pickle 

# Class which default operation unpacks to a list
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Superclass
class PickleMixin:
    # Save function for the model definitions
    def save(self, filename):
        with open("{}.pkl".format(filename), 'wb') as f:
            # Dump all method parameters
            pickle.dump(self, f)

    # Load object info and return the information.
    @staticmethod
    def load(filename):
        with open("{}.pkl".format(filename), 'rb') as f:
            return pickle.load(f)

# Gives an image from a path
def image_from(path):
    if os.path.exists(path):
        return np.array(Image.open(path), dtype=np.int64)
    return None

# Returns an array of images depending on directory and limit
def images_from_dir(dirpath, limit=float('inf')):
    images = []
    # Iterate through all files in directory
    for i, filename in enumerate(os.listdir(dirpath)):
        images.append(image_from(os.path.join(dirpath, filename)))
        # Break on limit reached
        if i >= limit - 1:
            break
    return np.asarray(images)

# not much use
if __name__ == '__main__':
    img = image_from('face/1.pgm')
    print(img.shape)
