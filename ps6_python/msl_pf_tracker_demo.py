import cv2
import numpy as np
import time
from msl_pf_tracker import *

def msl_pf_tracker_demo(videofile, textfile, frames_to_save=[], infix='a',
                        play_video=True, num_particles=100, dimensions=2,
                        control=10, sim_std=20, alpha=0):

    cap = cv2.VideoCapture('input/' + videofile + '.avi')

    # retrieve first frame
    ret, frame = cap.read()  # get frame
    if not ret:
        print('Could not retrieve initial frame for initialization!')
        quit()

    # get dimensions of image
    search_space = np.array(frame.shape[:2])

    # load model coordinates file and retrieve model from first frame
    f = open('input/' + textfile + '.txt', 'r').read().split()
    s = {'x': float(f[0]), 'y': float(f[1]), 'w': float(f[2]), 'h': float(f[3])}
    miny = int(s['y']); maxy = int(miny + s['h'])
    minx = int(s['x']); maxx = int(minx + s['w'])
    model = frame[miny:maxy, minx:maxx]
    cv2.imwrite('output/ps6-' + infix +'-1.png', frame[miny:maxy, minx:maxx])

    # create tracker
    tracker = MSLPFTracker(model, search_space, num_particles, dimensions,
                           control, sim_std, alpha)

    count = 1  # a frame has already been retrieved
    save_count=0  # count of the highlighted frames that will be saved

    while cap.isOpened():
        start_time = time.time()

        # retrieve next frame and convert to rgb
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

        # track model in frame
        tracker.update(frame)

        # visualize particle filter results in colored frame
        tracker.visualize_filter(frame)

        # insert the tracking model on the top left corner of the frame
        frame[:model.shape[0], :model.shape[1]] = tracker.model

        # add delay in order to adjust frame rate to about 40Hz
        delay = int(25 - (time.time() - start_time))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        cv2.imshow(videofile, frame)  # Display the resulting frame

        # store frames 28, 84, 144
        if count in (frames_to_save):
            cv2.imwrite('output/ps6-' + infix +'-'+str(save_count+2)+'.png',
                        frame)
            save_count += 1

    cap.release()
    cv2.destroyAllWindows()
