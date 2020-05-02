import numpy as np
from PIL import Image
import os
import json
import pickle 

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class PickleMixin:
    def save(self, filename):
        with open("{}.pkl".format(filename), 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open("{}.pkl".format(filename), 'rb') as f:
            return pickle.load(f)

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


if __name__ == '__main__':
    img = image_from('face/1.pgm')
    print(img.shape)
