import cv2
import numpy as np
import matplotlib.pyplot as plt

#LESSON 1
'''
imread function reads the image from the file. loads the image.
the function will return a multidimensional numpy array of relative intensity of each pixel
'''


image = cv2.imread('test_image.jpg')


'''
#imshow renders the image
#name of the window opened and the image itself
cv2.imshow('result', image)
#timer to show the window
cv2.waitKey(0)
'''

#_________________________________________________________________

#LESSON2 - Finding Lane lines / Edge Detection
'''
#image is stored as an array of pixels 
# a pixel can be [0, 232, 245, 255] : 0 black , 255 White
#strong gradient : 0 -----> 255, small gradient: 0 ---> 15
#discontinue in intensity or sharp graident we can obtain edges
#if we convert the image to grey scale it becomes a smaller array and becomes easy to compute
'''

lane_image = np.copy(image)
#gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

#_________________________________________________________________

#LESSON3 Applying Gaussian Blur to the image / smoothened

'''
1. Convert image to grayscale
2. Reduce Noise
cv2.GaussianBlur(gray, (5, 5), 0) creates a 5X5 matrix to smoothened the image

'''
#blur = cv2.GaussianBlur(gray, (5, 5), 0)


#_________________________________________________________________

##LESSON4 canny function 
'''
cv2.Canny(image, low_threshold, high_theshold)
calculates gradient, convert to 5X5 Grey Scale 
if the graident is below low_theshold it dicards the change, 
if it is above the high_threshold than the function accepts it as edge pixel
if it is between than 
'''

#canny = cv2.Canny(blur, 50, 150)
#cv2.imshow('result', canny)
#cv2.waitKey(0)

#_________________________________________________________________

#LESSON 5/6 create a function for above lessons

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_sloped_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2),(y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
           
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis = 0) 
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average) 
    return np.array([left_line, right_line]) 


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

'''
by Matplot lip we get the image in x-y axiz and we can isolate the traingl
on lane where we want to focus
#canny = canny(lane_image)  
#plt.imshow(canny)
#plt.show()  

'''

'''
This rOi function takes an image calculates the height finds the triangle in the
image by given axis and creates an array of triangles and masks everything else with dark color
'''


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def region_of_interest(image):
    #number of rows to get the height
    height = image.shape[0] 
    
    polygons = np.array([
        [(200, height ), (1100, height ), (550, 250)]
        ])
    mask = np.zeros_like(image)
    ##masked image pixel will be 1's
    cv2.fillPoly(mask, polygons, 255)
    #bitwise and operation between two image pixels
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


'''
for photo
canny = canny(lane_image)  
cropped_image = region_of_interest(canny)

lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average_sloped_intercept(lane_image, lines)
line_image = display_lines(lane_image, averaged_lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
cv2.imshow("result", combo_image)
cv2.waitKey(0) 
'''
#_________________________________________________________________

#LESSON 7 finding lane line by Hough Transform, as straight line is represented by the equation y = mx+b

cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame= cap.read()
    canny_image = canny(frame)  
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_sloped_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image)
    cv2.waitKey(1) 

