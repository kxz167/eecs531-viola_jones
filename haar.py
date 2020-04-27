import numpy as np
from integral_image import calc_int
from numba import njit, jit, jitclass, types
import numba as nb

# spec = [
#     ('add_points', nb.typed.List.empty_list(item_type=nb.Tuple)),
#     ('sub_points', nb.typed.List.empty_list(item_type=nb.Tuple)),
#     # ('shape', nb),
#     # ('feature_type', types.string)
# ]

# @jitclass(spec)
class HaarFeature:
    """
    A class that represents a single Haar feature.
    Contains a list of rectangular regions to add over, and a list of rectangular regions to subtract over.
    All rectangles in these features have the same dimensions (contained in self.shape).
    Also contains the type of Haar feature (2h, 2v, 3h, 3v, 4).
    """

    def __init__(self, a, s, sh, t):
        self.add_points = a
        self.sub_points = s
        self.shape = sh
        self.feature_type = t

    def __repr__(self):
        return f"""HaarFeature:
        Add: {self.add_points.__repr__()}
        Subtract: {self.sub_points.__repr__()}
        Shape: {self.shape}
        Type: {self.feature_type}\n"""
    
    def score(self, integral_image):
        return score(integral_image, self.add_points, self.sub_points, self.shape)

# @jit
def score(integral_image, add_points, sub_points, shape):
    return sum([compute_region(integral_image, point, shape) for point in add_points]) \
            - sum([compute_region(integral_image, point, shape) for point in sub_points]) 

def compute_region(integral_image, point, shape):
    y, x = (point[1], point[0])
    height, width = shape[1], shape[0]
    ii = integral_image
    # A = point
    # D = (point[0]+shape[0], point[1]+shape[1])
    # B = (D[0], A[1])
    # C = (A[0], D[1])
    # copy = np.copy(integral_image)
    ii = np.pad(integral_image, ((1, 0), (1, 0)))
    # return copy[A] + copy[D] - copy[C] - copy[B]
    return ii[y+height][x+width] + ii[y][x] - (ii[y+height][x]+ii[y][x+width])

# @jit(nopython=False)
def haar_features(image):
    """
    Returns a list of Haar features for the given image.
    """

    img_height, img_width = image.shape

    features = []
    # Loop through rectangle dimensions, then through image indices
    for feat_width in range(1, img_width + 1):
        for feat_height in range(1, img_height + 1):
            for feat_x in range(0, img_width - feat_width + 1):
                for feat_y in range(0, img_height - feat_height + 1):

                    base = (feat_x, feat_y)
                    shape = (feat_width, feat_height)

                    # 2 rectangles, horizontal
                    if feat_x + (2 * feat_width) <= img_width:
                        right = (feat_x + feat_width, feat_y)
                        features.append(HaarFeature([right], [base], shape, '2h'))

                    # 2 rectangles, vertical
                    if feat_y + (2 * feat_height) <= img_height:
                        bottom = (feat_x, feat_y + feat_height)
                        features.append(HaarFeature([bottom], [base], shape, '2v'))

                    # 3 rectangles, horizontal
                    if feat_x + (3 * feat_width) <= img_width:
                        middle = (feat_x + feat_width, feat_y)
                        right = (feat_x + (2 * feat_width), feat_y)
                        features.append(HaarFeature([middle], [base, right], shape, '3h'))

                    # 3 rectangles, vertical
                    if feat_y + (3 * feat_height) <= img_height:
                        middle = (feat_x, feat_y + feat_height)
                        bottom = (feat_x, feat_y + (2 * feat_height))
                        features.append(HaarFeature([middle], [base, bottom], shape, '3v'))

                    # 4 rectangles
                    if feat_x + (2 * feat_width) <= img_width and feat_y + (2 * feat_height) <= img_height:
                        top_right = (feat_x + feat_width, feat_y)
                        bottom_left = (feat_x, feat_y + feat_height)
                        bottom_right = (feat_x + feat_width, feat_y + feat_height)
                        features.append(HaarFeature([top_right, bottom_left], [base, bottom_right], shape, '4'))

    return features


if __name__ == '__main__':
    # Test
    integral = np.ones((3,3))
    hf = haar_features(integral)
    print(len(hf))
    integral = calc_int(np.ones((3,3)))
    hf = haar_features(integral)
    print(hf)
    print(hf[12].score(integral))
    print(integral)
    print(integral[0, 2])
    print(integral[0, 1])
