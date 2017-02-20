import cv2
import numpy as np
import matplotlib.pyplot as plt

def vis_optic_flow(flow):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros(flow.shape[:2]+(3,), dtype=np.uint8)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_vis

def vis_optic_flow_arrows(img, flow, filename, show=True):
    x = np.arange(0, img.shape[1], 1)
    y = np.arange(0, img.shape[0], 1)
    x, y = np.meshgrid(x, y)
    plt.figure()
    fig = plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    step = img.shape[0] / 50
    plt.quiver(x[::step, ::step], y[::step, ::step],
               flow[::step, ::step, 0], flow[::step, ::step, 1],
               color='r', pivot='middle', headwidth=2, headlength=3)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    if show:
        plt.show()

def vis_optic_flow_arrows_multi(images, flows, filename, show=False):
    # split the images in equal number of rows and cols
    cols = 2
    rows = np.ceil(float(len(flows)) / cols)

    # plot all images
    plt.figure(1)
    for idx, (img, flow) in enumerate(zip(images, flows)):
        plot_idx = rows * 100 + cols * 10 + 1 + idx
        plt.subplot(plot_idx)
        plt.imshow(img, cmap='gray', interpolation='bicubic')
        step=img.shape[0] / 15
        x = np.arange(0, img.shape[1], 1)
        y = np.arange(0, img.shape[0], 1)
        x, y = np.meshgrid(x, y)
        plt.quiver(x[::step, ::step], y[::step, ::step],
                   flow[::step, ::step, 0], flow[::step, ::step, 1],
                   color='r', pivot='middle', headwidth=5, headlength=5)

    # save plot to file
    plt.savefig(filename)

    # if show is set, display the figure
    if show:
        plt.show()
    plt.clf()
