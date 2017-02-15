import cv2
import numpy as np
from harris_values import *
from calc_grad import *

def harris_values(img, window_size=5, harris_scoring=0.04, norm=False):
    # calculate image gradients on x and y dimensions
    Ix = calc_grad_x(img, 3, 3, 0)
    Iy = calc_grad_y(img, 3, 3, 0)
    Ixx = Ix ** 2
    Ixy = Ix * Iy
    Iyx = Iy * Ix
    Iyy = Iy ** 2
    # create the weight window matrix
    c = np.zeros((window_size,)*2, dtype=np.float32);
    c[window_size / 2, window_size / 2] = 1.0
    w = cv2.GaussianBlur(c, (window_size,)*2, 0)
    #  w = np.ones((window_size,)*2)
    # calculate the harris values for all pixels of the image
    Rs = np.zeros(img.shape, dtype=np.float32)
    for r in range(w.shape[0]/2, img.shape[0] - w.shape[0]/2):
        minr = max(0, r - w.shape[0]/2)
        maxr = min(img.shape[0], minr + w.shape[0])
        for c in range(w.shape[1]/2, img.shape[1] - w.shape[1]/2):
            minc = max(0, c - w.shape[1]/2)
            maxc = min(img.shape[1], minc + w.shape[1])
            wIxx = Ixx[minr:maxr, minc:maxc]
            wIxy = Ixy[minr:maxr, minc:maxc]
            wIyx = Iyx[minr:maxr, minc:maxc]
            wIyy = Iyy[minr:maxr, minc:maxc]
            Mxx = (w * wIxx).sum()
            Mxy = (w * wIxy).sum()
            Myx = (w * wIyx).sum()
            Myy = (w * wIyy).sum()
            M = np.array([Mxx, Mxy, Myx, Myy]).reshape((2,2))
            Rs[r,c] = np.linalg.det(M)- harris_scoring * (M.trace() ** 2)
    if norm:
        Rs = cv2.normalize(Rs, Rs, alpha=0, beta=255,
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return Rs

def harris_corners(img, window_size=5, harris_scoring=0.04, threshold=1e-2,
                  nms_size=10):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # calculate harris values for all valid pixels
    Rs = harris_values(img, window_size, harris_scoring)
    # apply thresholding
    Rs = Rs * (Rs > (threshold * Rs.max())) * (Rs > 0)
    # apply non maximal suppression
    rows, columns = np.nonzero(Rs)
    new_Rs = np.zeros(Rs.shape)
    for r,c in zip(rows,columns):
        minr = max(0, r - nms_size / 2)
        maxr = min(img.shape[0], minr + nms_size)
        minc = max(0, c - nms_size / 2)
        maxc = min(img.shape[1], minc + nms_size)
        if Rs[r,c] == Rs[minr:maxr,minc:maxc].max():
            new_Rs[r,c] = Rs[r,c]
            #  Rs[minr:r, minc:c] = 0
            #  Rs[r+1:maxr, c+1:maxc] = 0
    return new_Rs
