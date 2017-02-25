import cv2
import numpy as np

def create_mhi_seq(binary_seq, tau=0.5, t_end=30):
    Mt = np.zeros(binary_seq[0].shape, dtype=np.float)

    for t, Bt in enumerate(binary_seq):
        Mt = tau * (Bt == 1) + np.clip(np.subtract(Mt, np.ones(Mt.shape)), 0,
                                       255) * (Bt == 0)
        if t == t_end:
            break

    return Mt.astype(np.uint8)

def create_binary_seq(videofile, num_frames=10, theta=127, blur_ksize=(3,3),
                      blur_sigma=1, open_ksize=(3,3)):
    cap = cv2.VideoCapture(videofile)
    binary_seq = []
    ret, frame_old = cap.read()

    if not ret:
        print('Failed to retrieve frame!')
        exit(0)

    frame_old = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)
    frame_old = cv2.GaussianBlur(frame_old, blur_ksize, blur_sigma)
    open_kernel = np.ones(open_ksize, dtype=np.uint8)

    for i in range(num_frames):
        ret, frame_new = cap.read()

        if not ret:
            print('Failed to retrieve frame!')
            break
        frame_new = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)
        frame_new = cv2.GaussianBlur(frame_new, blur_ksize, blur_sigma)
        binary_img = np.abs(cv2.subtract(frame_new, frame_old)) >= theta
        binary_img = binary_img.astype(np.uint8)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, open_kernel)
        #  binary_img = cv2.dilate(binary_img, np.ones((5,5)), iterations=3)
        binary_seq.append(binary_img)
        frame_old = frame_new

    return binary_seq
