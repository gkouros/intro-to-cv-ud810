import cv2
import numpy as np
from get_matches import *

def draw_keypoints(img1, img2):
    _,_,kpts1, kpts2, matches = get_matches(img1, img2)
    matched_image = np.array([])
    matched_image = cv2.drawMatches(img1, kpts1, img2, kpts2, matches[:10],
                                    flags=2, outImg=matched_image)
    return matched_image
