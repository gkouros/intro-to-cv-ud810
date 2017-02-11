import numpy as np
import cv2

def disparity_ncorr(L, R, block_size=5, disparity_range=30, lambda_factor=0):
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
            # find best match of template in strip
            error = cv2.matchTemplate(R_strip, tpl, method=cv2.TM_CCORR_NORMED)
            c_tf = max(c-rc_min-tpl_cols/2, 0)
            dist = np.arange(error.shape[1]) - c_tf
            cost = error - np.abs(dist * lambda_factor)
            _,_,_,max_loc = cv2.minMaxLoc(cost)
            D_L[r, c] = dist[max_loc[0]]
    return D_L
