import cv2
from matplotlib import pyplot as plt
import numpy as np

def detect_shape(contour):
    """Advanced function to detect the shape of a contour"""
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    num_vertices = len(approx)
    
    if num_vertices == 3:
        return 'triangle', num_vertices
    elif num_vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        return 'rectangle' if 0.9 <= aspect_ratio <= 1.1 else 'rectangle', num_vertices
    elif num_vertices == 5:
        return 'pentagon', num_vertices
    elif num_vertices == 8:
        return 'octagon', num_vertices
    elif num_vertices > 8:
        return 'circle', num_vertices
    else:
        return 'unknown', num_vertices
    
class Result:
    def __init__(self, score, ctr=np.array([[0,0]]).reshape((-1,1,2)).astype(np.int32)):
        self.score = score
        self.ctr = ctr
        if self.score != 0:
            area = cv2.contourArea(self.ctr)
            perimeter = cv2.arcLength(self.ctr, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity >= 0.8:
                self.shape="circle"
                self.num_vertices=99
                return
            approx = cv2.approxPolyDP(self.ctr, 0.015 * perimeter, True)
            self.num_vertices = len(approx)
            match self.num_vertices:
                case 0:
                    self.shape="unknown"
                case 3:
                    self.shape='triangle'
                case 4:
                    self.shape='rectangle'
                case 5:
                    self.shape='pentagon'
                case 6:
                    self.shape='hexagon'
                case 7:
                    self.shape='heptagon'
                case 8:
                    self.shape='octagon'
                case _:
                    self.shape='circle'
    def getEdgeNum(self) -> int:
        if self.score == 0:
            return 0
        return self.num_vertices
    def getScore(self) -> int:
        return self.score
    def __str__(self):
        return(f'Shape{self.shape}\nEdge Num:{self.num_vertices}')

def init_gui(filename,image,blurred,gray):
    # To visualize the results
    plt.figure(figsize=(20, 12))
    # Original image
    plt.subplot(3, 4, 1)
    plt.title(f'1. Original Image {filename}')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    # Grayscale
    plt.subplot(3, 4, 2)
    plt.title('2. Grayscale')
    plt.imshow(gray, cmap='gray')
    plt.axis('off')
    # Blurred
    plt.subplot(3, 4, 3)
    plt.title('3. Gaussian Blurred')
    plt.imshow(blurred, cmap='gray')
    plt.axis('off')
