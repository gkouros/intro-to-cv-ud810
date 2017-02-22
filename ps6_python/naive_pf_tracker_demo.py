import cv2
import numpy as np
import time
from naive_pf_tracker import *

def naive_pf_tracker_demo(videofile, frames_to_save=[], infix='a',
                          play_video=True, num_particles=100, dimensions=2,
                          control=10, noise=10, sim_std=20):
    cap = cv2.VideoCapture('input/' + videofile + '.avi')

    # retrieve first frame
    ret, frame = cap.read()  # get frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert frame to rgb
    search_space = np.array(gray.shape)
    if not ret:
        print('Could not retrieve initial frame for initialization!')

    # load model coordinates file and retrieve model from first frame
    f = open('input/' + videofile + '.txt', 'r').read().split()
    s = {'x': float(f[0]), 'y': float(f[1]), 'w': float(f[2]), 'h': float(f[3])}
    miny = int(s['y']); maxy = int(miny + s['h'])
    minx = int(s['x']); maxx = int(minx + s['w'])
    model = gray[miny:maxy, minx:maxx]
    cv2.imwrite('output/ps6-1-' + infix +'-1.png', frame[miny:maxy, minx:maxx])

    # create tracker
    tracker = NaivePFTracker(model, search_space, num_particles, dimensions,
                             control, noise, sim_std)

    count = 1  # a frame has already been retrieved
    save_count=0  # count of the highlighted frames that will be saved
    while cap.isOpened():
        start_time = time.time()

        # retrieve next frame and convert to rgb
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # track model in frame
        tracker.resample(gray)

        # visualize particle filter results in colored frame
        tracker.visualize_filter(frame)

        # add delay in order to adjust frame rate to about 40Hz
        delay = int(25 - (time.time() - start_time))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        cv2.imshow(videofile, frame)  # Display the resulting frame

        # store frames 28, 84, 144
        if count in (frames_to_save):
            cv2.imwrite('output/ps6-1-' + infix +'-'+str(save_count+2)+'.png',
                        frame)
            save_count += 1

    cap.release()
    cv2.destroyAllWindows()
