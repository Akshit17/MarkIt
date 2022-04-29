# https://stackoverflow.com/questions/72061864/error-finding-corner-points-of-a-rectangle-contour-with-cv-boundingrect

# I'm trying to extract the corner points of a rectangular section containing bubbles from an OMR sheet so I can later use those points to Wrap perspective to get bird's eye view on that section but I am not getting expected results. 

# Following is the OMR sheet image :- [OMRsheet.jpg][1]

# Code :-

import cv2 
import numpy as np

def extract_rect(contours):                         #Function to extract rectangular contours above a certain area unit  
    rect_contours = []
    for c in contours:
        if  cv2.contourArea(c) > 10000:
            perimeter = cv2.arcLength(c, True)  
            approx = cv2.approxPolyDP(c, 0.02*perimeter, True)    #approximates a curve or a polygon with another curve/polygon with less vertices so that the distance between them is less or equal to the specified precision. Uses Douglas-Peucker algorithm 
            if len(approx) == 4:
                rect_contours.append(c)

    rect_contours = sorted(rect_contours, key=cv2.contourArea,reverse=True)   # Sorting the contours based on area from large to small

    return rect_contours

def rect_points(rect_contour):                           #Function to find corner points of the contour passed          #Something wrong with this Not giving expected results. Messing up the wrapping of the image
    perimeter = cv2.arcLength(rect_contour, True)  
    approx = cv2.approxPolyDP(rect_contour, 0.02*perimeter, True)

    print("APPROX")
    print(type(approx))
    print(approx)
    cv2.drawContours(img, approx, -1, (100,10,55), 18)           #Rechecking if cotour passed to this function is the correct one
    cv2.drawContours(img, rect_contour, -1, (100,10,55), 1)
    

    x, y, w, h = cv2.boundingRect(rect_contour)         #I Suspect Logical error in this line as it returns corner points for the outer rectangle instead of the contour passed to it
    print("printing x y w h")
    print(x, y, w, h)

    # Corner points of the rectangle further used to be used to wrap the rectangular section
    point_1 = np.array([x, y])
    point_2 = np.array([x+w, y])
    point_3 = np.array([x, y+h])
    point_4 = np.array([w, h])

    corner_list = np.ndarray(shape=(4,2), dtype=np.int32)    
    np.append(corner_list, point_1)
    np.append(corner_list, point_2)
    np.append(corner_list, point_3)
    np.append(corner_list, point_4)

    print("corners list")
    print(corner_list)

    myPointsNew = np.zeros((4, 1, 2), np.int32)        
    add = corner_list.sum(1)
    # print(add)
    # print(np.argmax(add)) 
    myPointsNew[0] = corner_list[np.argmin(add)]   #[0,0]        #Setting up points in a coordinate system
    myPointsNew[3] = corner_list[np.argmax(add)]   #[w,h]
    diff = np.diff(corner_list, axis=1)
    myPointsNew[1] = corner_list[np.argmin(diff)]  #[w,0]
    myPointsNew[2] = corner_list[np.argmax(diff)]  #[h,0]

    print("mypointsnew")
    print(myPointsNew.shape)

    return myPointsNew

img_path = 'OMRsheet.jpg'
img = cv2.imread(img_path)

img_width = 700
img_height = 700
img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)



img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     
img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)      # blurred image
                        
img_canny = cv2.Canny(img_blur, 20, 110)                         # Edge detection on processed image using Canny edge detection , binary thresholding could have been an alternative (i.e  If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum value. )
        
contours, heirarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   # Find contours. 
                            #parameters are (input_image, retrieval_mode, approximation_method)
            
img_contours = img.copy()
cv2.drawContours(img_contours, contours, -1, (0,255,0), 1)  #parameters are (image, contours, countour_idx, contour_color, contour_thickness) . contour_idx is -1 for all contours
cv2.imshow('Contours', img_contours)

rect_contours = extract_rect(contours)
cv2.drawContours(img, rect_contours[1], -1, (0,255,0), 1)


rect_2 = rect_points(rect_contours[1])
cv2.drawContours(img, rect_2, -1, (0,0,255), 12)

warp_img_width = int(img_width/1.2)
warp_img_height = int(img_height/1.2)

warp_from = np.float32(rect_2)
warp_to = np.float32([[0,0], [warp_img_width, 0], [0, warp_img_height], [warp_img_width, warp_img_height]])
transformation_matrix = cv2.getPerspectiveTransform(warp_from, warp_to)
img_warp = cv2.warpPerspective(img, transformation_matrix, (warp_img_height, warp_img_height))
cv2.imshow('Wrapped Perspective', img_warp)

cv2.imshow('Original', img)

cv2.waitKey(0)

# Output for `cv2.imshow('Original', img)` :- [OMRsheet_contours.jpg][2]

# Output for `cv2.imshow('Wrapped Perspective', img_warp)` :-[Bird's Eye perspective.jpg][3]

# EXPECTED Output for `cv2.imshow('Wrapped Perspective', img_warp)` :- [Expected Bird's eye.jpg][4]

# Instead of getting wrapped perspective of the section containing only bubbles I am getting wrapped perspective for the whole paper which means either the points returned by `rect_points` function or the contour passed to the function i.e `rect_contours[1]` must have a mistake. The latter seemed to be fine as suggested after drawing contour lines for the contour passed to `rect_points` function. I suspect `x, y, w, h = cv2.boundingRect(rect_contour)` is returning incorrect points.
 
# Any idea on how I could solve this problem and get the [Expected Bird's eye.jpg][4] ?


#   [1]: https://i.stack.imgur.com/bcXv6.jpg
#   [2]: https://i.stack.imgur.com/i4Ska.png
#   [3]: https://i.stack.imgur.com/ofrrc.png
#   [4]: https://i.stack.imgur.com/LXZXx.png