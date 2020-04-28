import cv2 # opencv 사용
import numpy as np

def grayscale(img): # 흑백이미지로 변환
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold): # Canny 알고리즘
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size): # 가우시안 필터
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices, color3=(255,255,255), color1=255): # ROI 셋팅

    mask = np.zeros_like(img) # mask = img와 같은 크기의 빈 이미지
    
    if len(img.shape) > 2: # Color 이미지(3채널)라면 :
        color = color3
    else: # 흑백 이미지(1채널)라면 :
        color = color1
        
    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움 
    cv2.fillPoly(mask, vertices, color)
    
    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def draw_lines(img, lines, color=[0, 0, 255], thickness=2): # 선 그리기
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap): # 허프 변환
"""
image = 8bit. 즉 1채널인 흑백이미지를 넣어야함. 보통 Canny를 통해 edge를 찾은 후에 이 함수를 적용하므로 이미 흑백으로 변환된 상태

rho = hough space에서 ρ값을 한번에 얼만큼 증가시키면서 조사할 것인지를 묻는 것이다.
즉 rho = 1이라고 하면 1씩 증가시키면서 조사하겠다는 뜻이 된다. 보통 1을 넣는다.

theta = 단위는 라디안이다. 따라서 보통 각도 값을 입력한 후에 pi/180 을 곱해서 라디안 값으로 변환 시킨다.
각도 기준으로 [0:180] 사이의 범위로 입력하면 된다. 180도를 넘는 순간부터 직선이 unique 해지지 않기 때문이다.
(점1에서 점2로 그은 직선과 점2에서 점1로 그은 직선을 다른 직선으로 보는게 이상하지 않은가.)
여기서의 theta 또한 한번에 얼만큼 증가시키면서 조사할 것인지를 묻는 것이므로 보통 1도를 넣는다. 
즉 1도 * pi/180 라디안을 넣는다는 소리다.

threshold = 좀 전 이미지를 보면 Hough Space에서 교차점이 있지 않은가?
Hough Transform에서는 이 교차점이 하나씩 증가할때마다 +1을 해준다.
즉 위 이미지에서는 한 좌표에 교차가 9번있었으니 값이 9이다. 그런 식으로 그 값을 누적하고, 
나중에 그 누적값이 threshold 값을 넘는다면 직선이라고 판단하기 위해서 사용한다. 
이 말을 Image Space에서 바꿔말하면 서로 일직선 위에 있는 점의 수가 threshold 갯수 이상인지 아닌지를 판단하는 척도와 같은 말이다.
결국 threshold 값이  작으면 그만큼 기준이 낮아져 많은 직선이 검출될 것이고, 값을 높게 정하면 그만큼 적지만 확실한 직선들만 검출이 될 것이다.

output은 검출된 직선 만큼의 ρ와 θ이다. 
"""
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return line_img

def weighted_img(img, initial_img, α=1, β=1., λ=0.): # 두 이미지 operlap 하기
    return cv2.addWeighted(initial_img, α, img, β, λ)

image = cv2.imread('solidWhiteCurve.jpg') # 이미지 읽기
height, width = image.shape[:2] # 이미지 높이, 너비

gray_img = grayscale(image) # 흑백이미지로 변환
    
blur_img = gaussian_blur(gray_img, 3) # Blur 효과
        
canny_img = canny(blur_img, 70, 210) # Canny edge 알고리즘

vertices = np.array([[(50,height),(width/2-45, height/2+60), (width/2+45, height/2+60), (width-50,height)]], dtype=np.int32)
ROI_img = region_of_interest(canny_img, vertices) # ROI 설정

hough_img = hough_lines(ROI_img, 1, 1 * np.pi/180, 30, 10, 20) # 허프 변환

result = weighted_img(hough_img, image) # 원본 이미지에 검출된 선 overlap
cv2.imshow('result',result) # 결과 이미지 출력
cv2.waitKey(0) 
