import cv2
import numpy as np

def calc_grad_x(img, k_sobel=3, k_gauss=3, s_gauss=0, norm=False):
    grad_x = cv2.GaussianBlur(cv2.Sobel(img, cv2.CV_64F, 1, 0, k_sobel),
                              (k_gauss,)*2, s_gauss)
    if norm:
        grad_x = cv2.normalize(grad_x, grad_x, alpha=0, beta=255,
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return grad_x

def calc_grad_y(img, k_sobel=3, k_gauss=3, s_gauss=0, norm=False):
    grad_y = cv2.GaussianBlur(cv2.Sobel(img, cv2.CV_64F, 0, 1, k_sobel),
                              (k_gauss,)*2, s_gauss)
    if norm:
        grad_y = cv2.normalize(grad_y, grad_y, alpha=0, beta=255,
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return grad_y

def calc_grad_orientation(Ix, Iy):
    return np.arctan2(Iy, Ix)

def calc_grad_mag(Ix, Iy):
    return np.sqrt(Ix ** 2 + Iy ** 2)
