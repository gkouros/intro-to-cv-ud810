import cv2
import numpy as np

def calc_hu_moments(img):
    pq = [[2,0], [0,2], [1,2], [2,1], [2,2], [3,0], [0,3]]
    M_00 = img.sum()
    M_01 = np.sum(np.arange(img.shape[0]).reshape((-1,1)) * img)
    M_10 = np.sum(np.arange(img.shape[1]) * img)
    x_mean = M_10 / M_00
    y_mean = M_01 / M_00

    mu = np.zeros(len(pq))
    eta = np.zeros(len(pq))
    for idx,(p,q) in enumerate(pq):
        cx = (np.arange(img.shape[1]) - x_mean) ** p
        cy = ((np.arange(img.shape[0]) - y_mean) ** q).reshape((-1,1))
        mu[idx] = np.sum(cy * cx * img)
        eta[idx] = mu[idx] / img.sum() ** (1+(p+q)/2)

    return mu, eta

def calc_all_hu_moments(MHIs, MEIs):
    mu_list = []
    eta_list = []

    for MHI, MEI in zip(MHIs, MEIs):
        mu1, eta1 = calc_hu_moments(MHI)
        mu2, eta2 = calc_hu_moments(MEI)
        mu_list.append(np.append(mu1, mu2))
        eta_list.append(np.append(eta1, eta2))

    return mu_list, eta_list
