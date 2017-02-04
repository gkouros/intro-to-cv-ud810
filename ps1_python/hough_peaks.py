import numpy as np
import cv2

def hough_peaks(H, numpeaks=1, threshold=100, nhood_size=5):
    peaks = np.zeros((numpeaks,2), dtype=np.uint64)
    temp_H = H.copy()
    for i in range(numpeaks):
        _,max_val,_,max_loc = cv2.minMaxLoc(temp_H) # find maximum peak
        if max_val > threshold:
            peaks[i,0:2] = max_loc
            (c,r) = max_loc
            t = nhood_size
            # clear neighbourhood
            temp_H[int(r-t/2):int(r+t/2+1), int(c-t/2):int(c+t/2+1)] = 0
        else:
            peaks = peaks[:i]
            break
    return peaks
