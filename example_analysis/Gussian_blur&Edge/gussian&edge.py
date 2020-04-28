import cv2 # opencv 사용
import numpy as np

def grayscale(img): # 흑백이미지로 변환
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold): # Canny 알고리즘
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size): # 가우시안 필터
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

image = cv2.imread('solidWhiteCurve.jpg') # 이미지 읽기
height, width = image.shape[:2] # 이미지 높이, 너비

gray_img = grayscale(image) # 흑백이미지로 변환
    
blur_img = gaussian_blur(gray_img, 3) # Blur 효과 # 홀수만 가능하고 수가 클수록 edge가 적어짐
        
canny_img = canny(blur_img, 70, 210) # Canny edge 알고리즘 # 1:2나 1:3 비율을 추천함

cv2.imshow('result',canny_img) # Canny 이미지 출력
cv2.waitKey(0) 
