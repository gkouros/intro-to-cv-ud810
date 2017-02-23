import cv2
import numpy as np

def compare_hist(img1, img2, std=10, num_bins=8):
    if np.subtract(img1.shape, img2.shape).any():
        return 0.0
    else:
        x = chisqr(img1, img2, num_bins)
        return np.exp(-x / 2)


def chisqr(img1, img2, num_bins=8):
        hist1 = np.zeros(1*num_bins, dtype=np.float32)
        hist2 = np.zeros(1*num_bins, dtype=np.float32)
        K = num_bins
        for i in range(1):
            hist1[i*K:i*K+K] = cv2.calcHist(img1, [i], None, [num_bins], [0,256]).T
            hist2[i*K:i*K+K] = cv2.calcHist(img2, [i], None, [num_bins], [0,256]).T
            hist1[i*K:i*K+K] /= hist1[i*K:i*K+K].sum()
            hist2[i*K:i*K+K] /= hist2[i*K:i*K+K].sum()

        c = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        #  c = 0.5 * np.sum([((a - b) ** 2) / (a + b + 1e-10)
                          #  for (a, b) in zip(hist1, hist2)])
        return c

