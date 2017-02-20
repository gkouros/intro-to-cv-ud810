import cv2
import numpy as np

def laplacian_pyramid(img, levels):
    gpyr = gaussian_pyramid(img, levels)
    lpyr = [gpyr[levels-1]]
    for l in range(levels-1, 0, -1):
        G1 = expand(gpyr[l])
        G2 = gpyr[l-1]

        rdiff, cdiff = G2.shape[0] - G1.shape[0], G2.shape[1] - G1.shape[1]
        # if expanded image is bigger then crop it
        if rdiff < 0:
            G1 = G1[:G2.shape[0], :]
            rdiff = 0
        if cdiff < 0:
            G1 = G1[:, :G2.shape[1]]
            cdiff = 0
        # if expanded image is smaller then replicate border
        G1 = cv2.copyMakeBorder(G1, 0, rdiff, 0, cdiff, cv2.BORDER_REPLICATE)

        # subtract the two gaussian levels to get the corresponding laplacian
        L = cv2.subtract(G2, G1)
        lpyr.append(L)

    return lpyr[::-1]

def gaussian_pyramid(img, levels):
    pyr = [img]
    for l in range(levels-1):
        pyr.append(reduce(pyr[-1]))
    return pyr

def reduce(img):
    #  g = cv2.blur(img, (5,5))[::2, ::2]
    #  g = cv2.GaussianBlur(img, (3,3), 0)[::2, ::2]
    return cv2.pyrDown(img)

def expand(img):
    #  new_shape = (img.shape[1] * 2, img.shape[0] * 2)
    #  return cv2.GaussianBlur(cv2.resize(img, new_shape), (3,3), 0, cv2.INTER_AREA)
    return cv2.pyrUp(img)
