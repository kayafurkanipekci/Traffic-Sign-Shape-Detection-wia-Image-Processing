import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from common import detect_shape

def classifyByLargest(input_folder, output_folder):
    """classify traffic symbols by largest area method"""
    
    shapes = ['triangle', 'circle', 'rectangle', 'octagon', 'unknown']
    for shape in shapes:
        os.makedirs(os.path.join(output_folder, shape), exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue
        
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Image Couldn't Read: {filename}")
            continue
        
        # Processing steps
        # Tried to use different thresholding methods
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        methods = [
            ('Otsu Binary', cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
            ('Adaptive Gaussian', cv2.adaptiveThreshold(blurred, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)),
            ('Adaptive Mean', cv2.adaptiveThreshold(blurred, 255, 
                                cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2)),
            ('Canny', cv2.Canny(blurred, 50, 200))
        ]
        
        # To visualize the results
        plt.figure(figsize=(20, 12))
        
        # Original image
        plt.subplot(3, 4, 1)
        plt.title('1. Original Image')
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
        
        largest_area = 0
        best_shape = None
        best_vertice = 0
        best_method_name = None
        best_contour = None
        scores = {'Canny':0,'Otsu Binary':0,'Adaptive Gaussian':0,'Adaptive Mean':0}
        vertices = {'Canny':0,'Otsu Binary':0,'Adaptive Gaussian':0,'Adaptive Mean':0}
        # Loop over each thresholding method
        for idx, (method_name, thresh) in enumerate(methods, start=1):
            edges = cv2.Canny(thresh, 50, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
            if contours:
                for contour in contours:
                    area = cv2.contourArea(contour)
                    shape, vertice = detect_shape(contour)
                    if scores[method_name] < area: 
                        vertices[method_name]=vertice
                        scores[method_name]=area
                    if area > largest_area:
                        largest_area = area
                        best_shape = shape
                        best_vertice = vertice
                        best_method_name = method_name
                        best_contour = contour
            # Thresholding results
            if idx == 1:
                plt.subplot(3, 4, 4)
            else:
                plt.subplot(3, 4, idx + 3)
            plt.title(f'4.{idx}. {method_name}')
            plt.imshow(thresh, cmap='gray')
            plt.axis('off')
            
            # Edge detection results
            plt.subplot(3, 4, idx + 7)
            plt.title(f'5.{idx}. Edge Detection ({method_name})\n Edge Numbers:{vertices[method_name]}\n {scores[method_name]}')
            plt.imshow(edges, cmap='gray')
            plt.axis('off')
            
            # Tried to find the best contour for largest area method
        
        if best_contour is None:
            print(f"Couldn't Find Contour: {filename}")
            continue
        
        # Last result
        img_with_contours = image.copy()
        cv2.drawContours(img_with_contours, [best_contour], -1, (0,255,0), 3)
        
        plt.subplot(3, 4, 12)
        plt.title(f'6. Final Result\nShape: {best_shape}\nEdge Number: {best_vertice}\nMethod: {best_method_name}')
        plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Copy the image to the related folder
        output_path = os.path.join(output_folder, best_shape, filename)
        shutil.copy(image_path, output_path)
    
    print("Classification Completed!")

