import numpy as np
import cv2

def hough_lines_acc(img, rho_res=1, thetas=np.arange(-90,90,1)):
    rho_max = int(np.linalg.norm(img.shape-np.array([1,1]), 2));
    rhos = np.arange(-rho_max, rho_max, rho_res)
    thetas -= min(min(thetas),0)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)
    yis, xis = np.nonzero(img) # use only edge points
    for idx in range(len(xis)):
        x = xis[idx]
        y = yis[idx]
        temp_rhos = x * np.cos(np.deg2rad(thetas)) + y * np.sin(np.deg2rad(thetas))
        temp_rhos = temp_rhos / rho_res + rho_max
        m, n = accumulator.shape
        valid_idxs = np.nonzero((temp_rhos < m) & (thetas < n))
        temp_rhos = temp_rhos[valid_idxs]
        temp_thetas = thetas[valid_idxs]
        c = np.stack([temp_rhos,temp_thetas], 1)
        cc = np.ascontiguousarray(c).view(np.dtype((np.void, c.dtype.itemsize * c.shape[1])))
        _,idxs,counts = np.unique(cc, return_index=True, return_counts=True)
        uc = c[idxs].astype(np.uint)
        accumulator[uc[:,0], uc[:,1]] += counts.astype(np.uint)
    accumulator = cv2.normalize(accumulator, accumulator, 0, 255,
                                cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return accumulator, thetas, rhos
