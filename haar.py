import numpy as np


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
    
    def score(self, integral_image, pad=False):
        return score(integral_image, self.add_points, self.sub_points, self.shape, pad=pad)

@njit
def score(integral_image, add_points, sub_points, shape, pad=False):
    # add = [compute_region(integral_image, point, shape, pad) for point in add_points]
    add, sub = 0, 0
    for point in add_points:
        add += compute_region(integral_image, point, shape, pad)
    for point in sub_points:
        sub += compute_region(integral_image, point, shape, pad)
    return add - sub

@njit
def compute_region(integral_image, point, shape, pad=False):
    y, x = (point[1], point[0])
    height, width = shape[1], shape[0]
    # Padding is not numba compatible
    # if pad:
    #     # continue ii = np.pad(integral_image, ((1, 0), (1, 0)))
    #     pass
    # return integral_image[y+height][x+width] + integral_image[y][x] - (integral_image[y+height][x] + integral_image[y][x+width])
    return ii_value(integral_image, x+width, y+height) + ii_value(integral_image, x, y) - (ii_value(integral_image, x, y+height) + ii_value(integral_image, x+width, y))

@njit
def ii_value(integral_image, x, y):
    if x == 0 or y == 0:
        return 0
    return integral_image[y-1][x-1]

# @jit(nopython=True)
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
    integral = np.array([[1, 3, 4], [2, 8, 10], [43, 45, 67]])
    hf = haar_features(integral)
    print(hf)
