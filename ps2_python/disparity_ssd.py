import numpy as np
import cv2
from numpy.lib.stride_tricks import as_strided

def my_ssd(x, t):
    tc = t.shape[1]
    xc = x.shape[1]
    s = np.array([np.sum((t - x[:,i-tc/2:i+tc/2+1]) ** 2)
                  for i in range(tc / 2, xc - tc/2)], dtype=np.float32)
    return s[None,:]

def sumsqdiff2(input_image, template, valid_mask=None):
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


def disparity_ssd(L, R, block_size=5, disparity_range=30, lambda_factor=0):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))

    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """

    im_rows, im_cols = L.shape
    tpl_rows = tpl_cols = block_size
    D_L = np.zeros(L.shape, dtype=np.float32)


    for r in range(tpl_rows/2, im_rows-tpl_rows/2):
        tr_min, tr_max = max(r-tpl_rows/2, 0), min(r+tpl_rows/2+1, im_rows)
        for c in range(tpl_cols/2, im_cols-tpl_cols/2):
            # get template
            tc_min = max(c-tpl_cols/2, 0)
            tc_max = min(c+tpl_cols/2+1, im_cols)
            tpl = L[tr_min:tr_max, tc_min:tc_max].astype(np.float32)
            # get R strip in a window with width=disparity_range
            rc_min = max(c-disparity_range/2, 0)
            rc_max = min(c+disparity_range/2+1, im_cols)
            R_strip = R[tr_min:tr_max, rc_min:rc_max].astype(np.float32)
            #  error = my_ssd(R_strip, tpl)  # slow
            #  error = sumsqdiff2(R_strip, tpl) # faster
            error = cv2.matchTemplate(R_strip, tpl, method=cv2.TM_SQDIFF_NORMED) # fastest
            c_tf = max(c-rc_min-tpl_cols/2, 0)
            dist = np.arange(error.shape[1]) - c_tf
            cost = error + np.abs(dist * lambda_factor)
            _,_,min_loc,_ = cv2.minMaxLoc(cost)
            D_L[r, c] = dist[min_loc[0]]
    return D_L
