#! /usr/local/bin/python3

import sys

import cv2
import numpy as np
 
cap = cv2.VideoCapture(0)
 
while(1):
    _, frame = cap.read()
    # It converts the BGR color space of image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hues = {
        "green": {
            "lower": 40,
            "upper": 70
        },

        "purple": {
            "lower": 157,
            "upper": 165
        }
    }

    lower_saturation = 20
    upper_saturation = 150

    lower_value = 30
    upper_value = 200

    masks = []

    for key in hues:
        hue = hues[key]

        lower = np.array([hue["lower"], lower_saturation, lower_value])
        upper = np.array([hue["upper"], upper_saturation, upper_value])

        mask = cv2.inRange(hsv, lower, upper)

        masks.append(mask)

    combined_masks = cv2.bitwise_or(masks[0], masks[1])

    result = cv2.bitwise_and(frame, frame, mask = combined_masks)
 
    contours, hierarchy = cv2.findContours(combined_masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if w < 150 or h < 150:
            continue

        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('result', result)
     
    if cv2.waitKey(15) & 0xff == ord('q'):
        break
 
cv2.destroyAllWindows()
cap.release()
