import cv2
import numpy as np

def hough_circles_acc(edge_img, radius):
    accumulator = np.zeros(edge_img.shape, dtype=np.uint8)
    yis, xis = np.nonzero(edge_img) # coordinates of edges
    num_px = len(xis)
    (m,n) = edge_img.shape
    for x,y in zip(xis,yis):
        theta = np.arange(0,360)
        a = (y - radius * np.sin(theta * np.pi / 180)).astype(np.uint)
        b = (x - radius * np.cos(theta * np.pi / 180)).astype(np.uint)
        valid_idxs = np.nonzero((a < m) & (b < n))
        a, b = a[valid_idxs], b[valid_idxs]
        c = np.stack([a,b], 1)
        cc = np.ascontiguousarray(c).view(np.dtype((np.void, c.dtype.itemsize * c.shape[1])))
        _,idxs,counts = np.unique(cc, return_index=True, return_counts=True)
        uc = c[idxs]
        accumulator[uc[:,0], uc[:,1]] += counts.astype(np.uint)
    return accumulator
