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
            "lower": 150,
            "upper": 170
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
    warped = frame[:]
 
    contours, hierarchy = cv2.findContours(combined_masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        min_rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(min_rect)
        outline = np.int0(box)
        
        width = int(min_rect[1][0])
        height = int(min_rect[1][1])

        if width < 150 or height < 150:
            continue

        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")

        transform = cv2.getPerspectiveTransform(src_pts, dst_pts)

        roi = frame[y:y+h, x:x+w]
        warped = cv2.warpPerspective(frame, transform, (width, height))

        cv2.drawContours(result, [outline], 0, (0, 0, 255), 3)
        cv2.drawContours(result, [contour], 0, (0, 255, 0), 3)
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('result', result)
    cv2.imshow('warped', warped)
     
    if cv2.waitKey(15) & 0xff == ord('q'):
        break
 
cv2.destroyAllWindows()
cap.release()
