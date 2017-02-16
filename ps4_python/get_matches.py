import cv2
import numpy as np
from get_keypoints import *

def get_matches(img1, img2):
    img1_feat, kpts1 = get_keypoints(img1, draw_keypoints=False)
    img2_feat, kpts2 = get_keypoints(img2, draw_keypoints=False)

    # create sift instance
    sift = cv2.xfeatures2d.SIFT_create()
    # get descriptors
    descriptors1 = sift.compute(img1, kpts1)[1]
    descriptors2 = sift.compute(img2, kpts2)[1]

    # get matches
    bfm = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bfm.match(descriptors1, descriptors2)
    matches = sorted(matches, key = lambda x:x.distance)

    return img1_feat, img2_feat, kpts1, kpts2, matches
