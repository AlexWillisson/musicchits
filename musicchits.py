#! /usr/local/bin/python3

import sys

import cv2
import numpy as np
 
cap = cv2.VideoCapture(0)
 
while(1):
    _, frame = cap.read()
    # It converts the BGR color space of image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     
    # Threshold of blue in HSV space
    lower_green = np.array([40, 20, 30])
    upper_green = np.array([70, 150, 200])
 
    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_green, upper_green)
     
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-green regions
    result = cv2.bitwise_and(frame, frame, mask = mask)
 
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    # cv2.imshow('mask', mask)
    cv2.imshow('result', result)
     
    if cv2.waitKey(15) & 0xff == ord('q'):
        break
 
cv2.destroyAllWindows()
cap.release()
