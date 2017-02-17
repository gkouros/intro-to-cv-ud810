import cv2
import numpy as np
from harris_corners import *

def get_keypoints(img, draw_keypoints=True):
    if len(img) > 1:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # find harris corners -> keypoints
    # !!! Uses opencv builtin detector, since it's faster than mine
    corners = cv2.cornerHarris(gray,2,3,0.04)
    corners = cv2.normalize(corners, corners, alpha=0, beta=255,
                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    threshold=85
    rows, cols = np.nonzero(corners > threshold)
    # calculate the image gradients
    Ix = calc_grad_x(gray)
    Iy = calc_grad_y(gray)
    O = calc_grad_orientation(Ix, Iy)
    Mag = calc_grad_mag(Ix, Iy)
    # assign the keypoints
    keypoints = np.zeros((len(rows),))
    keypoints = []
    for i in range(len(rows)):
        r = rows[i]; c = cols[i]
        kp = cv2.KeyPoint(c, r, _size=10, _angle=np.rad2deg(O[r,c]), _octave=0)
        keypoints.append(kp)
    # draw the keypoints on the image
    if draw_keypoints:
        cv2.drawKeypoints(img, keypoints, img,
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img, keypoints
