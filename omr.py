from charset_normalizer import detect
import cv2
import numpy as np

#TODO: Abstract the rectangular area containing the bubbles from the image

def extract_rect(contours):                          #Needs optimization as it does not work for all bubblesheet examples
    rect_contours = []
    for c in contours:
        if  cv2.contourArea(c) > 10000:
            perimeter = cv2.arcLength(c, True)  
            approx = cv2.approxPolyDP(c, 0.02*perimeter, True)    #approximates a curve or a polygon with another curve/polygon with less vertices so that the distance between them is less or equal to the specified precision. Uses Douglas-Peucker algorithm 
            # if len(approx) == 4:               #cv2.boundingRect seems to be automatically taking care of this
            #     rect_contours.append(c)
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
    print(type(approx))
    print(approx)
    cv2.drawContours(img, approx, -1, (100,10,55), 18)
    cv2.drawContours(img, rect_contour, -1, (100,10,55), 1)
   

    x, y, w, h = cv2.boundingRect(rect_contour)
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

    new_cornerlist = np.zeros((4, 1, 2), np.int32) 
    add = corner_list.sum(1)
    # print(add)
    # print(np.argmax(add))
    new_cornerlist[0] = corner_list[np.argmin(add)]   #[0,0]
    new_cornerlist[3] = corner_list[np.argmax(add)]   #[w,h]
    diff = np.diff(corner_list, axis=1)
    new_cornerlist[1] = corner_list[np.argmin(diff)]  #[w,0]
    new_cornerlist[2] = corner_list[np.argmax(diff)]  #[h,0]

    print(new_cornerlist.shape)

    return new_cornerlist


#TODO: Find the bubbles and draw them on the image
def find_bubbles(img):
    rows = np.vsplit(img,10)              #Only work for evenly spead out bubbles
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,10)
        for box in cols:
            boxes.append(box)
            cv2.imshow("Boxes",rows[0])
    return boxes


def hough_circles(img):

    #load image
    # img = self.imager.values[self.imager.index, :, :]
    # img = np.float(self.image)
    image8 = np.uint8(img)
    output = image8.copy()

    #apply hough transform
    circles = cv2.HoughCircles(image8, cv2.HOUGH_GRADIENT, 24, 50)

    #place circles and cente rectangle on image
    if circles is not None:
        circles = np.round(circles[0, :].astype("int"))

        for (x, y, r) in circles:
            cv2.circle(output, (x,y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        cv2.imshow("output", np.hstack([image8, output]))
        cv2.waitKey(0)

def detect_blob(image):
        # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()
    
    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 90
    
    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.2
    
    # Set Convexity filtering parameters
    params.filterByConvexity = True
    params.minConvexity = 0.2
        
    # Set inertia filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
        
    # Detect blobs
    keypoints = detector.detect(image)
    
    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    number_of_blobs = len(keypoints)
    text = "Number of Circular Blobs: " + str(len(keypoints))
    cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    
    # Show blobs
    cv2.imshow("Filtering Circular Blobs Only", blobs)
    cv2.waitKey(0)

# img_path = 'bubblesheet_1.jpg'
# img = cv2.imread(img_path)

# img_path = 'bubblesheet_2.jpg'
# img = cv2.imread(img_path)

# img_path = 'bubblesheet_3.jpg'
# img = cv2.imread(img_path)

img_path = 'bubblesheet_4.jpg'
img = cv2.imread(img_path)


print(img.shape)            # Original size is 1600 * 1200 and 3 color channels
img_width = 600
img_height = 600
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

rect_contours = extract_rect(contours)
cv2.drawContours(img, rect_contours[1], -1, (0,255,0), 1)

wrap_points = rect_points(rect_contours[2])

if wrap_points.size != 0:
    cv2.drawContours(img, wrap_points, -1, (0,0,255), 12)

print("how both contour same")
print(rect_contours[0])
print("contour 2")
print(rect_contours[1])

wrap_points = rect_points(rect_contours[2])
cv2.drawContours(img, wrap_points, -1, (0,0,255), 12)


# warp_img_width = int(img_width/2)
# warp_img_height = int(img_height/2)
warp_img_width = int(img_width)
warp_img_height = int(img_height)


warp_from = np.float32(wrap_points)
warp_to = np.float32([[0,0], [warp_img_width, 0], [0, warp_img_height], [warp_img_width, warp_img_height]])
transformation_matrix = cv2.getPerspectiveTransform(warp_from, warp_to)
img_warp = cv2.warpPerspective(img, transformation_matrix, (warp_img_height, warp_img_height))
cv2.imshow('Wrapped Perspective', img_warp)

img_warp_gray = cv2.cvtColor(img_warp, cv2.COLOR_BGR2GRAY)
# img_thresh  = cv2.threshold(img_warp_gray, 120, 255, cv2.THRESH_BINARY_INV)[1]

#Using adaptive threshold to get the best threshold value
img_thresh = cv2.adaptiveThreshold(img_warp_gray, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
cv2.imshow('img_thresh', img_thresh)

# find_bubbles(img_thresh)
# hough_circles(img_thresh)
detect_blob(img_thresh)

print(img_thresh.shape)

cv2.imshow('Original', img)


#TODO: Implement logic for marks counting

cv2.waitKey(0)



