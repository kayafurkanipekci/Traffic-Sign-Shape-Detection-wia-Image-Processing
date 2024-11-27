import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections.abc import Mapping

import common 

def classifyImages(input_folder, output_folder):
    """classify traffic symbols by best quality methods"""
    
    shapes = ['triangle', 'circle', 'rectangle', 'octagon', 'unknown']
    for shape in shapes:
        path = os.path.join(output_folder, shape)
        os.makedirs(path, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Image Couldn't Read: {filename}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray,(3, 3), sigmaX=0, sigmaY=0)
        common.init_gui(filename,image,blurred,gray)

        # preprocessing
        methods = [
            ('Otsu Binary', cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ('Adaptive Gaussian', cv2.adaptiveThreshold(blurred, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)),
            ('Adaptive Mean', cv2.adaptiveThreshold(blurred, 255, 
                                cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2)),
            ('No-Threshold', blurred)
        ]
        idx = 0
        results: Mapping[str,common.Result] = {}
        for (thresh_name, thresh) in methods:
            edges = cv2.Canny(thresh, 60, 180)
            ctrs, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            ctrs = [cnt for cnt in ctrs if cv2.contourArea(cnt) > 100]
            #return best contours else return 0 0 contour
            if ctrs: 
                for ctr in ctrs:
                    quality_score = evaluate_contour_quality(ctr, image.shape[:2])
                    if not thresh_name in results:
                        results[thresh_name] = common.Result(thresh_name,quality_score,ctr)
                    elif results[thresh_name].score <= quality_score:
                        results[thresh_name] = common.Result(thresh_name,quality_score,ctr)
            else:
                results[thresh_name] = common.Result(thresh_name, 0)
            plt.subplot(3,4,5+idx)
            idx+=1
            plt.title(f'4.{idx}. {thresh_name}')
            plt.imshow(thresh, cmap='gray')
            plt.axis('off')
            plt.subplot(3, 4, 8+idx)
            plt.title(f'5.{idx}. Canny:({thresh_name})\n{results[thresh_name]}')
            plt.imshow(edges, cmap='gray')
            plt.axis('off')
        
        result_img = image.copy()
        best_contour:str = max(results, key=lambda k: results[k].getScore())

        cv2.drawContours(result_img, results[best_contour].ctr, -1, (0,255,0), 2)
        plt.subplot(3,4,4)
        plt.title(f'6. Final Result\n Canny:({best_contour})\n Shape:{results[best_contour].shape}')
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.tight_layout()        
        plt.show()
        print(f'{filename}\n')

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
    size_score = 1 - abs(0.4 - area_ratio) if area_ratio <= 0.9 else 0
    if circularity >= 0.8:
        circularity*=2
    # Total score calculation (We can adjust the weights)
    total_score = (0.3 * circularity + 
                  0.3 * center_score + 
                  0.4 * size_score)
    return total_score