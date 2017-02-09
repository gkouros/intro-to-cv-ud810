import numpy as np
import cv2
from numpy.lib.stride_tricks import as_strided

def my_ssd(t, x):
    tc = t.shape[1]
    xc = x.shape[1]
    s = np.array([np.sum((t - x[:,i:i+tc]) ** 2) for i in range(xc - tc + 1)],
                 dtype=np.float32)
    return s[None,:]

def sumsqdiff2(template, input_image, valid_mask=None):
    '''
    ssd implementation borrowed from:
        http://stackoverflow.com/questions/17881489/faster-way-to-calculate-sum-of-squared-difference-between-an-image-m-n-and-a
    '''
    if valid_mask is None:
        valid_mask = np.ones_like(template)
    total_weight = valid_mask.sum()
    window_size = template.shape
    # Create a 4-D array y, such that y[i,j,:,:] is the 2-D window
    #     input_image[i:i+window_size[0], j:j+window_size[1]]
    y = as_strided(input_image,
                    shape=(input_image.shape[0] - window_size[0] + 1,
                           input_image.shape[1] - window_size[1] + 1,) +
                          window_size,
                    strides=input_image.strides * 2)
    # Compute the sum of squared differences using broadcasting.
    ssd = ((y - template) ** 2 * valid_mask).sum(axis=-1).sum(axis=-1)
    return ssd


def disparity_ssd(L, R, template_size=3):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))

    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """

    im_rows = L.shape[0]; im_cols = L.shape[1]
    tpl_rows = tpl_cols = template_size
    D_L = np.ndarray((im_rows - tpl_rows + 1, im_cols - tpl_cols + 1),
                     dtype=np.float32)
    for r in range(D_L.shape[0]):
        for c in range(D_L.shape[1]):
            tpl = L[r:r+tpl_rows, c:c+tpl_cols].astype(np.float32)
            R_strip = R[r:r+tpl_rows, :].astype(np.float32)
            #  res = my_ssd(tpl, R_strip)
            #  res = sumsqdiff2(tpl, R_strip)
            res = cv2.matchTemplate(R_strip, tpl, method=cv2.TM_SQDIFF)
            _,_,min_loc,_ = cv2.minMaxLoc(res)
            D_L[r, c] = min_loc[0] - c
    return D_L


