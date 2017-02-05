import cv2
import numpy as np

def hough_lines_draw(img, outfile, peaks, rhos, thetas):
    for peak in peaks:
        rho = rhos[peak[0]]
        theta = thetas[peak[1]] * np.pi / 180.0
        a = np.cos(theta); b = np.sin(theta)
        pt0 = rho * np.array([a,b])
        pt1 = tuple((pt0 + 1000 * np.array([-b,a])).astype(int))
        pt2 = tuple((pt0 - 1000 * np.array([-b,a])).astype(int))
        cv2.line(img, pt1, pt2, (0,255,0), 2)
    cv2.imwrite(outfile, img)
