import cv2
import numpy as np

def similarity(img1, img2, std=10):
    if np.subtract(img1.shape, img2.shape).any():
        return 0
    else:
        mse = np.sum(np.subtract(img1, img2, dtype=np.float32) ** 2)
        mse /= float(img1.shape[0] * img1.shape[1])
        return np.exp(-mse / 2 / std**2)
