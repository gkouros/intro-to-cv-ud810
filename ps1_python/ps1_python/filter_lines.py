import cv2
import numpy as np

def filter_lines(peaks, thetas, rhos, theta_threshold, rho_threshold):
    del_list = []
    for i in range(len(peaks)):
        delta_rho = np.abs(np.array([abs(rhos[peaks[j,0]] - rhos[peaks[i,0]])
                              for j in range(len(peaks))]))
        delta_theta = np.array([abs(thetas[peaks[j,1]] - thetas[peaks[i,1]])
                                for j in range(len(peaks))])
        if not ((delta_theta < theta_threshold) & (delta_rho > 1) &
                (delta_rho < rho_threshold)).any():
            del_list += [i]

    peaks = np.delete(peaks, del_list, 0)
    return peaks
