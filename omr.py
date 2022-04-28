import cv2
import numpy as np

#TODO: Abstract the rectangular area containing the bubbles from the image

def extract_rect(contours):                          #Needs optimization as it does not work for all bubblesheet examples
    rect_contours = []
    for c in contours:
        if  cv2.contourArea(c) > 10000:
            perimeter = cv2.arcLength(c, True)  
            approx = cv2.approxPolyDP(c, 0.02*perimeter, True)    #approximates a curve or a polygon with another curve/polygon with less vertices so that the distance between them is less or equal to the specified precision. Uses Douglas-Peucker algorithm 
            if len(approx) == 4:
                rect_contours.append(c)

    rect_contours = sorted(rect_contours, key=cv2.contourArea,reverse=True)

    return rect_contours


def findlen(p1, p2):
    length = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5       # length = sqroot[(x2-x1)**2 + (y2-y1)**2]
    
    return length

def rect_points(rect_contour):                                     #Something wrong with this Not giving expected results. Messing up the wrapping of the image
    perimeter = cv2.arcLength(rect_contour, True)  
    approx = cv2.approxPolyDP(rect_contour, 0.02*perimeter, True)

    print("APPROX")
    print(approx)
    cv2.drawContours(img, approx, -1, (100,10,55), 18)

    x, y, w, h = cv2.boundingRect(approx)
    print("printing x y w h")
    print(x, y, w, h)

    # Corner points of the rectangle further used to be used to wrap the rectangular section
    point_1 = np.array([x, y])
    point_2 = np.array([x+w, y])
    point_3 = np.array([x, y+h])
    point_4 = np.array([w, h])

    paper_length = findlen(point_1,  point_3)
    paper_breadth = findlen(point_1,  point_2)

    if paper_breadth > paper_length: # here breadth should be smaller than length 
        temp = point_1.copy()
        point_1 = point_3.copy()
        point_3 = point_4.copy()
        point_4 = point_2.copy()
        point_2 = temp

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
    myPointsNew[0] = corner_list[np.argmin(add)]   #[0,0]
    myPointsNew[3] = corner_list[np.argmax(add)]   #[w,h]
    diff = np.diff(corner_list, axis=1)
    myPointsNew[1] = corner_list[np.argmin(diff)]  #[w,0]
    myPointsNew[2] = corner_list[np.argmax(diff)]  #[h,0]

    print("mypointsnew")
    print(myPointsNew.shape)

    return myPointsNew

    print(corner_list)
    print(corner_list.shape)

    return corner_list

def rect_points_2(rect_contour):                                   #Trial 
    perimeter = cv2.arcLength(rect_contour, True)  
    approx = cv2.approxPolyDP(rect_contour, 0.02*perimeter, True)
    print(approx)
    print(approx.shape)

    approx = approx.reshape((4, 2)) # REMOVE EXTRA BRACKET
    print(approx)
    myPointsNew = np.zeros((4, 1, 2), np.int32) # NEW MATRIX WITH ARRANGED POINTS
    add = approx.sum(1)
    print(add)
    print(np.argmax(add))
    myPointsNew[0] = approx[np.argmin(add)]  #[0,0]
    myPointsNew[3] =approx[np.argmax(add)]   #[w,h]
    diff = np.diff(approx, axis=1)
    myPointsNew[1] =approx[np.argmin(diff)]  #[w,0]
    myPointsNew[2] = approx[np.argmax(diff)] #[h,0]

    print(myPointsNew)

    return myPointsNew

    # return approx


#TODO: Find the bubbles and draw them on the image
def find_bubbles(contours):
    rows = np.vsplit(img,5)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,5)
        for box in cols:
            boxes.append(box)
    return boxes


# img_path = 'bubblesheet_1.jpg'
# img = cv2.imread(img_path)

# img_path = 'bubblesheet_2.jpg'
# img = cv2.imread(img_path)

# img_path = 'bubblesheet_3.jpg'
# img = cv2.imread(img_path)

img_path = 'bubblesheet_4.jpg'
img = cv2.imread(img_path)


print(img.shape)            # Original size is 1600 * 1200 and 3 color channels
img_width = 700
img_height = 700
img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)



img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # grayscaled image            Grayscaling bcoz further for edge detection we only require intensity info., and similar intensity pixels to detect the contours
img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)      # blurred image
                        # (inputimage, kernel size , sigma_x)          when only sigma_x is provived sigma_y is taken as the same as sigma_x  (sigma_x is Gaussian kernel standard deviation in X direction)
                        #https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
cv2.imshow('Blurred', img_blur)

img_canny = cv2.Canny(img_blur, 20, 110)                         # Edge detection on processed image using Canny edge detection , binary thresholding could have been an alternative (i.e  If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum value. )
                 #  ( input_img, Lower Threshold, Upper threshold)       
                    #  https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html

cv2.imshow('Edge detection', img_canny)

contours, heirarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   # Find contours. 
                            #parameters are (input_image, retrieval_mode, approximation_method)
           
img_contours = img.copy()
cv2.drawContours(img_contours, contours, -1, (0,255,0), 1)  #parameters are (image, contours, countour_idx, contour_color, contour_thickness) . contour_idx is -1 for all contours
cv2.imshow('Contours', img_contours)

img_thresh  = cv2.threshold(img_gray, 170, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow('Threshold', img_thresh)

rect_contours = extract_rect(contours)
cv2.drawContours(img, rect_contours[1], -1, (0,255,0), 1)

# rect_1 = rect_points(rect_contours[0])
# rect_2 = rect_points(rect_contours[1])

# if rect_1.size != 0 and rect_2.size != 0:
#     print("Entered successfully")
#     cv2.drawContours(img, rect_1, -1, (255,0,0), 12)
#     cv2.drawContours(img, rect_2, -1, (0,0,255), 12)

print("how both contour same")
print(rect_contours[0])
print("contour 2")
print(rect_contours[1])

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


#TODO: Implement logic for marks counting

cv2.waitKey(0)



