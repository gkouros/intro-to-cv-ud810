import cv2
import numpy as np
import random
from get_matches import *

def ransac_trans(transA, transB):
    img1, img2, kpts1, kpts2, matches = get_matches(transA, transB)
    tolerance = 10
    best_match = 0
    consensus_set = []
    # find consensus of translation between a random keypoint and the rest for
    # a number of times to find the best match regarding translation
    for i in range(100):
        idx = random.randint(0, len(matches)-1)
        kp1 = kpts1[matches[idx].queryIdx]
        kp2 = kpts2[matches[idx].trainIdx]
        dx = int(kp1.pt[0] - kp2.pt[0])
        dy = int(kp1.pt[1] - kp2.pt[1])
        temp_consensus_set = []
        for j, match in enumerate(matches):
            kp1 = kpts1[match.queryIdx]
            kp2 = kpts2[match.trainIdx]
            dxi = int(kp1.pt[0] - kp2.pt[0])
            dyi = int(kp1.pt[1] - kp2.pt[1])
            if abs(dx - dxi) < tolerance and abs(dy - dyi) < tolerance:
                temp_consensus_set.append(j)
        if len(temp_consensus_set) > len(consensus_set):
            consensus_set = temp_consensus_set
            best_match = idx
    # calculate best match translation
    kp1 = kpts1[matches[best_match].queryIdx]
    kp2 = kpts2[matches[best_match].trainIdx]
    dx = int(kp1.pt[0] - kp2.pt[0])
    dy = int(kp1.pt[1] - kp2.pt[1])
    consensus_matches = np.array(matches)[consensus_set]
    matched_image = np.array([])
    # draw matches with biggest consensus
    matched_image = cv2.drawMatches(img1, kpts1, img2, kpts2, consensus_matches,
                                    flags=2, outImg=matched_image)
    print('Best match: idx=%d with consensus=%d or %d%%\nTranslation: dx=%dpx '
          'and dy=%dpx'%(best_match, len(consensus_set),
                         100*len(consensus_set) / len(matches), dx, dy))
    return matched_image
