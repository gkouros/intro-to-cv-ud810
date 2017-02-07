import cv2
import numpy as np

img = cv2.imread('input/ps1-input2.png', cv2.IMREAD_COLOR)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 2, 55, param1=200, param2=80, minRadius=20, maxRadius=40)
output = img.copy()
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    cv2.imshow("output", np.hstack([img,output]))
    cv2.waitKey(0)

