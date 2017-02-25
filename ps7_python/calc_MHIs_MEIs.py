import cv2
import numpy as np
from frame_differenced_mhi import *

videofile = lambda a, p, t: 'input/PS7A'+str(a)+'P'+str(p)+'T'+str(t)+'.avi'

def calc_MHIs_MEIs(skip_person_idx=0):

    # binary sequence and MHI params for each person
    t_end = [70, 40, 50]
    theta = [3, 3, 20]
    tau = [60, 40, 50]

    # calculate MHIs for each person, action and trial
    MHIs = []
    labels = []
    for action in [1,2,3]:
        for person in [p for p in [1,2,3] if p != skip_person_idx]:
            for trial in [1,2,3]:
                # calculate the binary sequence
                binary_seq = create_binary_seq(videofile(action, person, trial),
                                               num_frames=t_end[person-1],
                                               theta=theta[person-1],
                                               blur_ksize=(85,)*2,
                                               blur_sigma=0, open_ksize=(9,)*2)
                # calculate the motion history image
                M_tau = create_mhi_seq(binary_seq, tau=tau[person-1],
                                       t_end=t_end[person-1]).astype(np.float)
                # normalize the motion history image
                cv2.normalize(M_tau, M_tau, 0.0, 255.0, cv2.NORM_MINMAX)
                MHIs.append(M_tau)
                labels.append(action)

    # calculate motion energy images by thresholding motion history images
    MEIs = [(255*M>0).astype(np.uint8) for M in MHIs]

    return MHIs, MEIs, labels
