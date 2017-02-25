# ps7
import cv2
import numpy as np
import sys
from collections import OrderedDict
import matplotlib.pyplot as plt
from frame_differenced_mhi import *
from hu_moments import *
from plot_confusion_matrix import *
from plot_nearest_neighbour_confusion import *
from calc_MHIs_MEIs import *

'''
Problem Set 7: Motion History Images
'''

videofile = lambda a, p, t: 'input/PS7A'+str(a)+'P'+str(p)+'T'+str(t)+'.avi'

# 1. Frame-Differenced MHI
# ===========================

def ps7_1_a():
    binary_seq = create_binary_seq(videofile(1,2,1), num_frames=31,
                                   theta=2, blur_ksize=(55,)*2, blur_sigma=0,
                                   open_ksize=(9,)*2)
    for idx in [10,20,30]:
        bf = binary_seq[idx]
        cv2.normalize(bf, bf, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite('output/ps7-1-a-'+str(idx/10)+'.png', bf)

def ps7_1_b():
    t_end = [35, 30, 30]
    thetas = [4, 4, 4]
    taus = [40, 35, 30]
    for idx in range(1):
        binary_seq = create_binary_seq(videofile(1,2,idx+1), t_end[idx],
                                       thetas[idx], (85,)*2, 0, (9,)*2)
        M_tau = create_mhi_seq(binary_seq, taus[idx],
                               t_end[idx]).astype(np.float)
        cv2.normalize(M_tau, M_tau, 0.0, 255.0, cv2.NORM_MINMAX)
        cv2.imwrite('output/ps7-1-b-'+str(idx+1)+'.png', M_tau)


# 2. Recognition using MHIs
# ==========================

def ps7_2_a():
    # calculate the motion history/energy images for all actions,people,trials
    MHIs, MEIs, labels = calc_MHIs_MEIs()

    # calculate the hu-moments for all MHIs and corresponding MEIs
    mu_list, eta_list = calc_all_hu_moments(MHIs, MEIs)

    # train a k-NN classifier using the hu-moments from both MHIs and MEIs
    train_data = np.array(mu_list).astype(np.float32)
    train_data2 = np.array(eta_list).astype(np.float32)
    labels = np.array(labels).astype(np.int)
    plot_nearest_neighbour_confusion(train_data, labels,
                                     'output/ps7-2-a-1.png')
    plot_nearest_neighbour_confusion(train_data2, labels,
                                     'output/ps7-2-a-2.png')


def ps7_2_b():
    cnf_matrices = []
    for person in [1,2,3]:
        # calculate the motion history/energy images
        MHIs, MEIs, labels = calc_MHIs_MEIs(skip_person_idx=person)

        # calculate the hu-moments for all MHIs and corresponding MEIs
        _,eta_list = calc_all_hu_moments(MHIs, MEIs)

        # train a k-NN classifier using the hu-moments from both MHIs and MEIs
        train_data = np.array(eta_list).astype(np.float32)
        labels = np.array(labels).astype(np.int)
        cnf_matrix = plot_nearest_neighbour_confusion(
            train_data, labels, 'output/ps7-2-b-'+str(person)+'.png')
        cnf_matrices.append(cnf_matrix)

    cnf_matrix = np.sum(cnf_matrices, 0)
    plot_confusion_matrix(cnf_matrix, ['action1','action2','action3'],
                          normalize=True, filename='output/ps7-2-b-4.png')


ps7_list = OrderedDict([('1a', ps7_1_a), ('1b', ps7_1_b), ('2a', ps7_2_a),
                        ('2b', ps7_2_b)])

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] in ps7_list:
            print('Executing task %s'%sys.argv[1])
            ps7_list[sys.argv[1]]()
        else:
            print('\nGive argument from list \n%s\nfor the corresponding task.'
                  %ps7_list.keys())
    else:
        print('\n * Executing all tasks: * \n')
        for idx in range(len(ps7_list)):
            print('Executing task: %s'%
                  ps7_list.keys()[idx])
            ps7_list.values()[idx]()
