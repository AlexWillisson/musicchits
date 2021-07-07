#! /usr/local/bin/python3

import cv2
import numpy as np
 
cap = cv2.VideoCapture(0)
 
while(1):
    _, frame = cap.read()
    # It converts the BGR color space of image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     
    # Threshold of blue in HSV space
    lower_green = np.array([50, 30, 30])
    upper_green = np.array([70, 150, 200])
 
    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_green, upper_green)
     
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-green regions
    result = cv2.bitwise_and(frame, frame, mask = mask)
 
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)
     
    if cv2.waitKey(20) & 0xff == ord('q'):
        break
 
cv2.destroyAllWindows()
cap.release()
