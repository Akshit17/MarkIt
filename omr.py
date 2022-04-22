import cv2
import numpy as np


img_path = 'bubblesheet_1.jpg'
img = cv2.imread(img_path)

print(img.shape)            # Original size is 1600 * 1200 and 3 color channels
img_width = 500
img_height = 500
img = cv2.resize(img, (img_width, img_height))

# print(cv2.getGaussianKernel())

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # grayscaled image            Grayscaling bcoz further for edge detection we only require intensity info., and similar intensity pixels to detect the contours
img_blur = cv2.GaussianBlur(img_gray, (5,5), 1)      # blurred image
                        # (inputimage, kernel size , sigma_x)          when only sigma_x is provived sigma_y is taken as the same as sigma_x  (sigma_x is Gaussian kernel standard deviation in X direction)
                        #https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
cv2.imshow('Blurred', img_blur)

img_canny = cv2.Canny(img_blur, 100, 200)                         # Edge detection on processed image using Canny edge detection , binary thresholding could have been an alternative (i.e  If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum value. )
                 #  ( input_img, Lower Threshold, Upper threshold)
                    #  https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
print(img_canny)
cv2.imshow('Edge detection', img_canny)

contours = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   # Find contours. 
                            #parameters are (input_image, retrieval_mode, approximation_method)
print(contours)                             
img_contours = img.copy()
cv2.drawContours(img_contours, contours[0], -1, (0,255,0), 1)  #parameters are (image, contours, countour_idx, contour_color, contour_thickness) . contour_idx is -1 for all contours
cv2.imshow('Contours', img_contours)

cv2.imshow('Original', img)


#TODO: Find the bubbles and draw them on the image

#TODO: Implement logic for marks counting
cv2.waitKey(0)




# cap = cv2.VideoCapture(0)

# while True:
#     _, frame = cap.read()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # hue and sat value for hsv img
#     lower_red = np.array([0, 50, 25])   
#     upper_red = np.array([255, 255, 180])

#     mask = cv2.inRange(hsv, lower_red, upper_red)  # masking
#     cv2.imshow('mask', mask)

#     res = cv2.bitwise_and(frame, frame, mask=mask)  

#     smoothed = cv2.filter2D(res, -1, np.ones((15, 15), np.float32)/225)
#     cv2.imshow('smoothed', smoothed)

#     blur = cv2.GaussianBlur(res, (15, 15), 0)
#     cv2.imshow('blur', blur)

#     median = cv2.medianBlur(res, 15)

#     cv2.imshow('res', res)

#     cv2.imshow('frame', frame)

#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:      # key 27 is ESC
#         break
