import cv2
import numpy as np
from plot_confusion_matrix import *

def plot_nearest_neighbour_confusion(train_data, labels, filename):
    # train a k-NN classifier using the hu-moments from both MHIs and MEIs
    classifier = cv2.ml.KNearest_create()
    cnf_matrix = np.zeros((3,3))
    np.set_printoptions(precision=2)

    for i in range(len(train_data)):
        train_idxs = range(0, i) + range(i+1, train_data.shape[0])
        # train dataset
        X_train = train_data[train_idxs]
        y_train = labels[train_idxs]
        # test dataset
        X_test = np.array([train_data[i]])
        y_test = np.array([labels[i]])
        # train knn classifier
        classifier.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
        # predict test data labels
        _,results,_,_ = classifier.findNearest(X_test, 1)
        cnf_matrix[y_test-1, int(results[0])-1] += 1

    # plot confusion matrix based on classification results of k-NN
    plot_confusion_matrix(cnf_matrix, classes=['action1','action2','action3'],
                          normalize=True, title='Confusion matrix',
                          filename=filename)

    return cnf_matrix
