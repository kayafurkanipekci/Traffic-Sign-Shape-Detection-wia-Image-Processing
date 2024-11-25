import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import platform

def classifyByQuality(input_folder, output_folder):
    """classify traffic symbols by best quality methods"""
    clean_output_folder(output_folder)
    
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
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)
        
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
        best_vertices = 0
        best_method_name = None
        best_contour = None
        best_thresh = None
        best_edges = None
        best_score = 0
        
        # Loop over each thresholding method
        for idx, (method_name, thresh) in enumerate(methods, start=1):
            edges = cv2.Canny(thresh, 50, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
            
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
            plt.title(f'5.{idx}. Kenar Tespiti ({method_name})')
            plt.imshow(edges, cmap='gray')
            plt.axis('off')
            
            # Tried to find the best contour for best quality edges method
            if contours:
                for contour in contours:
                    quality_score = evaluate_contour_quality(contour, image.shape[:2])
                    shape, vertices = detect_shape(contour)
                    
                    if quality_score > best_score:
                        best_score = quality_score
                        best_shape = shape
                        best_vertices = vertices
                        best_method_name = method_name
                        best_contour = contour
        
        if best_contour is None:
            print(f"Couldn't Find Contour: {filename}")
            continue
        
        # Last result
        img_with_contours = image.copy()
        cv2.drawContours(img_with_contours, [best_contour], -1, (0,255,0), 3)
        
        plt.subplot(3, 4, 12)
        plt.title(f'6. Final Result\nShape: {best_shape}\nEdge Number: {best_vertices}\nMethod: {best_method_name}')
        plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Copy the image to the related folder
        output_path = os.path.join(output_folder, best_shape, filename)
        shutil.copy(image_path, output_path)
    
    print("Classification Completed!")

def evaluate_contour_quality(contour, image_shape):
    """Evaluate the quality of a contour based on its area and perimeter"""

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Regularity of shape (closer to 1 means more regular)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Center distance
    moments = cv2.moments(contour)
    if moments['m00'] != 0:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        center_dist = np.sqrt((cx - image_shape[1]/2)**2 + (cy - image_shape[0]/2)**2)
        center_score = 1 - (center_dist / (np.sqrt(image_shape[0]**2 + image_shape[1]**2)/2))
    else:
        center_score = 0
    
    # The size of the contour (not too small or too large)
    total_area = image_shape[0] * image_shape[1]
    area_ratio = area / total_area
    size_score = 1 - abs(0.3 - area_ratio) if area_ratio <= 0.8 else 0
    
    # Total score calculation (We can adjust the weights)
    total_score = (0.3 * circularity + 
                  0.3 * center_score + 
                  0.5 * size_score)
    
    return total_score
