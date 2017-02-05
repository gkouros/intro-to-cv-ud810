import numpy as np
import cv2

def hough_lines_acc(img, rho_res=1, thetas=np.arange(-90,90,1)):
    rho_max = int(np.linalg.norm(img.shape-np.array([1,1]), 2));
    rhos = np.arange(-rho_max, rho_max, rho_res)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)
    yis, xis = np.nonzero(img) # coordinates of edges
    for idx in range(len(xis)):
        x = xis[idx]
        y = yis[idx]
        for theta in thetas:
            rho = x * np.cos(np.deg2rad(theta)) + y * np.sin(np.deg2rad(theta))
            rho_idx = int(rho/rho_res + rho_max)
            theta_idx = theta - min(0, thetas[0])
            accumulator[rho_idx, theta_idx] += 1
    accumulator = cv2.normalize(accumulator, accumulator, 0, 255,
                                cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return accumulator, thetas, rhos
