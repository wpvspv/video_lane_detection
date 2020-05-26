from picamera.array import PiRGBArray
import RPi.GPIO as GPIO
from picamera import PiCamera
import time
import cv2
import numpy as np
import math

#GPIO.setmode(GPIO.BCM)
#GPIO.setup(13, GPIO.OUT)
#GPIO.setup(19, GPIO.OUT)
#GPIO.setup(5, GPIO.OUT)
#GPIO.setup(12, GPIO.OUT)

#GPIO.output(5, GPIO.LOW)
#GPIO.output(12, GPIO.LOW)
#GPIO.output(19, GPIO.LOW)
#GPIO.output(13, GPIO.LOW)

theta=0
minLineLength = 5
maxLineGap = 10

camera = PiCamera()
camera.resolution = (960, 540)
camera.framerate = 15

rawCapture = PiRGBArray(camera, size=(960, 540))
time.sleep(0.1)


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def detect_lanes_img(img):

    height, width = img.shape[:2]

    # Set ROI
    vertices = np.array([[(50,height),(width/2-45, height/2+60), (width/2+45, height/2+60), (width-50,height)]], dtype=np.int32)
    ROI_img = region_of_interest(img, vertices)
    
    # Convert to grayimage
    #g_img = grayscale(img)
       
    # Apply gaussian filter
    blur_img = gaussian_blur(ROI_img, 3)
        
    # Apply Canny edge transform
    canny_img = canny(blur_img, 70, 210)
    # to except contours of ROI image
    vertices2 = np.array([[(52,height),(width/2-43, height/2+62), (width/2+43, height/2+62), (width-52,height)]], dtype=np.int32)
    canny_img = region_of_interest(canny_img, vertices2)
    
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
   #GPIO.output(5, GPIO.LOW)
   #GPIO.output(12, GPIO.LOW)
   #GPIO.output(13, GPIO.LOW)
   #GPIO.output(19, GPIO.LOW)
   time.sleep(0.0)
   image = frame.array
   height, width = image.shape[:2]
   vertices = np.array([[(50,height),(width/2-45, height/2+60), (width/2+45, height/2+60), (width-50,height)]], dtype=np.int32)
   ROI_img = region_of_interest(image, vertices)   
   gray = grayscale(ROI_img)
   blurred = cv2.GaussianBlur(gray, (5, 5), 0)
   edged = cv2.Canny(blurred, 85, 85)
   vertices2 = np.array([[(52,height),(width/2-43, height/2+62), (width/2+43, height/2+62), (width-52,height)]], dtype=np.int32)
   edged = region_of_interest(edged, vertices2)
   
   lines = cv2.HoughLinesP(edged,1,np.pi/180,10,minLineLength,maxLineGap)
   
   if(lines !=[]):
       for x in range(0, len(lines)):
           for x1,y1,x2,y2 in lines[x]:
               cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
               theta=theta+math.atan2((y2-y1),(x2-x1))
          
   threshold=6
   
   if(theta>threshold):
   #    GPIO.output(5, GPIO.LOW)
   #    GPIO.output(12, GPIO.HIGH)
   #    GPIO.output(19, GPIO.LOW)
   #    GPIO.output(13, GPIO.HIGH)
       print("left")
       
   if(theta<-threshold):
   #    GPIO.output(12, GPIO.LOW)
   #    GPIO.output(5, GPIO.HIGH)
   #    GPIO.output(13, GPIO.LOW)
   #    GPIO.output(19, GPIO.HIGH)
       print("right")
       
   if(abs(theta)<threshold):
   #   GPIO.output(5, GPIO.LOW)
   #   GPIO.output(12, GPIO.HIGH)
   #   GPIO.output(13, GPIO.LOW)
   #   GPIO.output(19, GPIO.HIGH)
      print ("straight")
   
   theta=0
   cv2.imshow("Frame",image)
   key = cv2.waitKey(1) & 0xFF
   rawCapture.truncate(0)
   
   if key == ord("q"):
       break
