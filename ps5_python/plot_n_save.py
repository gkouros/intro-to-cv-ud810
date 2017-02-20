import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_n_save(filename, images, show=False):
    # split the images in equal number of rows and cols
    cols = 2
    rows = np.ceil(float(len(images)) / cols)

    # plot all images
    plt.figure(1)
    for idx, img in enumerate(images):
        plot_idx = rows * 100 + cols * 10 + 1 + idx
        plt.subplot(plot_idx)
        plt.imshow(img)

    # save plot to file
    plt.savefig(filename)

    # if show is set, display the figure
    if show:
        plt.show()
    plt.clf()
