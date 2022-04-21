import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # hue and sat value for hsv img
    lower_red = np.array([0, 50, 25])   
    upper_red = np.array([255, 255, 180])

    mask = cv2.inRange(hsv, lower_red, upper_red)  # masking
    cv2.imshow('mask', mask)

    res = cv2.bitwise_and(frame, frame, mask=mask)  
    cv2.imshow('res', res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:      # key 27 is ESC
        break







    # img_path = 'bubblesheet_1.jpg'
    # img = cv2.imread(img_path)

    # print(img.shape)            # Original size is 1600 * 1200 and 3 color channels
    # img_width = 500
    # img_height = 500
    # img = cv2.resize(img, (img_width, img_height))

    # print(cv2.getGaussianKernel())

    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # grayscaled image
    # img_blur = cv2.GaussianBlur(img_gray, (5,5), 1)      # blurred image
    #                         # (inputimage, kernel size , sigma_x)          when only sigma_x is provived sigma_y is taken as the same as sigma_x  (sigma_x is Gaussian kernel standard deviation in X direction)
    #                         #https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html



    # cv2.imshow('Original', img)
    # cv2.waitKey(0)