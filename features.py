'''
Define Haar like features

We have 5 types: [0, 1], [[1], [0]], [0, 1, 0], [[0], [1], [0]], 
                    [[0, 1], [1, 0]]
'''

import numpy as np
from integral_image import region_sum, calc_int
from typing import Tuple
from enum import Enum, auto

class FeatureType(Enum):
    '''
    Features 2 Vertical, 2 Horizontal, 3 Vertical, 3 Horizontal, Chessboard
    '''

    F2V = auto()
    F2H = auto() 
    F3V = auto() 
    F3H = auto() 
    F4C = auto()

class HaarFeature:

    '''
    Calculate Haar feature sum at the expected region
    '''
    def __init__(self, feature_type, region: Tuple[Tuple[int]]):
        assert len(region) == 2, 'Follow top left, bottom right points in an integral image'
        self.region = region
        self.feature_type = feature_type

    def score(self, integral_image):
        (A, D) = self.region
        if self.feature_type == FeatureType.F2V:
            '''
            A00B
            0000
            1111
            C11D
            '''
            midpoint_left = (A[0], A[1] + abs(A[1] - D[1])//2)
            midpoint_right = (D[0], A[1] + abs(A[1] - D[1])//2) 
            dark = region_sum(integral_image, (A, midpoint_right))
            white = region_sum(integral_image, (midpoint_left, D))
            return white - dark
        if self.feature_type == FeatureType.F2H:
            '''
            A10B
            1100
            1100
            C10D
            '''
            midpoint_top = (A[0] + abs(A[0] - D[0])//2, A[1])
            midpoint_bottom = (A[0] + abs(A[0] - D[0])//2, D[1])
            white = region_sum(integral_image, (A, midpoint_bottom))
            dark = region_sum(integral_image, (midpoint_top, D))
            return white - dark
        if self.feature_type == FeatureType.F3H:
            '''
            A0110B
            001100
            C0110D
            '''
            ones_left_bottom = (A[0] + abs(A[0] - D[0])//3, D[1])
            ones_left_top = (A[0] + abs(A[0] - D[0])//3, A[1])
            ones_right_bottom = (A[0] + 2*abs(A[0] - D[0])//3, D[1])
            ones_right_top = (A[0] + 2*abs(A[0] - D[0])//3, A[1])

            white_left = region_sum(integral_image, (A, ones_left_bottom))
            dark = region_sum(integral_image, (ones_left_top, ones_right_bottom))
            white_right = region_sum(integral_image, (ones_right_top, D))
            return white_left + white_right - dark
        if self.feature_type == FeatureType.F3V:
            '''
            A00B
            0000
            1111
            1111
            0000
            C00D
            '''
            ones_left_top = (A[0], A[1] + abs(A[1] - D[1])//3)
            ones_right_top = (D[0], A[1] + abs(A[1] - D[1])//3)
            ones_left_bottom = (A[0], A[1] + 2*abs(A[1] - D[1])//3)
            ones_right_bottom = (D[0], A[1] + 2*abs(A[1] - D[1])//3)
            white_top = region_sum(integral_image, (A, ones_right_top))
            dark = region_sum(integral_image, (ones_left_top, ones_right_bottom))
            white_bottom = region_sum(integral_image, (ones_left_bottom, D))
            return white_top + white_bottom - dark
        if self.feature_type == FeatureType.F4C:
            '''
            A01B
            0011
            1100
            C10D
            '''
            center_x = A[0] + abs(A[0] - D[0])//2
            center_y = A[1] + abs(A[1] - D[1])//2
            white_top_left = region_sum(integral_image, (A, (center_x, center_y)))
            white_bottom_right = region_sum(integral_image, ((center_x, center_y), D))
            dark_top_right = region_sum(integral_image, ((center_x, A[1]), (D[0], center_y)))
            dark_bottom_left = region_sum(integral_image, ((A[0], center_y), (center_x, D[1])))
            return white_top_left + white_bottom_right - (dark_bottom_left + dark_top_right)
        raise AttributeError(msg='State cannot be reached')



        

if __name__ == '__main__':
    # test code
    img = np.ones((4, 4))
    ii = calc_int(img)
    feature = HaarFeature(FeatureType.F2V, ((0, 0), (2, 2)))
    assert feature.score(ii) == 0, 'Amount of white and dark are equal, expect 0, return {}'.format(feature.score(ii))
    feature = HaarFeature(FeatureType.F2H, ((0, 0), (2, 2)))
    assert feature.score(ii) == 0, 'Amount of white and dark are equal, expect 0, return {}'.format(feature.score(ii))
    feature = HaarFeature(FeatureType.F4C, ((0, 0), (2, 2)))
    assert feature.score(ii) == 0, 'Amount of white and dark are equal, expect 0, return {}'.format(feature.score(ii))
    feature = HaarFeature(FeatureType.F3V, ((0, 0), (2, 3)))
    assert feature.score(ii) == 2, 'white has 4 pixels, dark have 2, expected is 2, result is {}'.format(feature.score(ii))
    feature = HaarFeature(FeatureType.F3H, ((0, 0), (3, 2)))
    assert feature.score(ii) == 2, 'white has 4 pixels, dark have 2, expected is 2, result is {}'.format(feature.score(ii))
