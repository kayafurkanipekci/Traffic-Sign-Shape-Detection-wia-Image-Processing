import cv2
import platform

# This is to test the images or others
image: cv2.typing.MatLike
if platform.system() == 'Windows':
    image = cv2.imread('traffic_Data\\DATA\\mix\\038_0001.png')
elif platform.system() == 'Linux':
    image = cv2.imread("traffic_Data/DATA/mix/038_0001.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), cv2.BORDER_DEFAULT)
canny = cv2.Canny(blur, 120, 255, 1)

corners = cv2.goodFeaturesToTrack(canny, 4, 0.5, 50)

if corners is not None:
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image, (int(x), int(y)), 5, (36, 255, 12), -1)

cv2.imshow('canny', canny)
cv2.imshow('image', image)
cv2.waitKey()
