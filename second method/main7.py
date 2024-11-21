import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import math


def detect_shape(contour):
    if contour is None or len(contour) < 3:
        return 'Belirsiz', 0

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
    
    num_vertices = len(approx)
    
    if num_vertices == 3:
        # Üçgen için ek kontrol
        angles = []
        for i in range(3):
            pt1 = approx[i][0]
            pt2 = approx[(i+1) % 3][0]
            pt3 = approx[(i+2) % 3][0]
            angle = abs(math.degrees(math.atan2(pt3[1]-pt2[1], pt3[0]-pt2[0]) - math.atan2(pt1[1]-pt2[1], pt1[0]-pt2[0])))
            angles.append(angle)
        
        # Üçgenin açılarını kontrol et
        if all(30 < angle < 150 for angle in angles):
            return 'Üçgen', num_vertices
    
    elif num_vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        return 'Kare' if 0.9 <= aspect_ratio <= 1.1 else 'Dikdörtgen', num_vertices
    elif num_vertices == 8:
        return 'Sekizgen', num_vertices
    elif num_vertices > 8:
        return 'Daire', num_vertices
    else:
        return 'Bilinmeyen', num_vertices
    
    
def process_and_visualize(image, filename):
    # Görüntü işleme adımları
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    methods = [
        ('Otsu Binary', cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
        ('Adaptive Gaussian', cv2.adaptiveThreshold(blurred, 255, 
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2)),
        ('Adaptive Mean', cv2.adaptiveThreshold(blurred, 255, 
                            cv2.ADAPTIVE_THRESH_MEAN_C, 
                            cv2.THRESH_BINARY_INV, 11, 2))
    ]
    
    plt.figure(figsize=(20, 15))
    
    # Orijinal görüntü
    plt.subplot(3, 3, 1)
    plt.title('Orijinal')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Gri tonlama
    plt.subplot(3, 3, 2)
    plt.title('Gri Tonlama')
    plt.imshow(gray, cmap='gray')
    plt.axis('off')
    
    # Bulanıklaştırma
    plt.subplot(3, 3, 3)
    plt.title('Bulanıklaştırma')
    plt.imshow(blurred, cmap='gray')
    plt.axis('off')
    
    best_shape = 'Belirsiz'
    best_vertices = 0
    best_method_name = None
    best_contour = None
    largest_area = 0
    
    for idx, (method_name, thresh) in enumerate(methods):
        # Eşikleme
        plt.subplot(3, 3, 4 + idx)
        plt.title(f'{method_name} Eşikleme')
        plt.imshow(thresh, cmap='gray')
        plt.axis('off')
        
        # Kenar tespiti
        edges = cv2.Canny(thresh, 50, 200)
        plt.subplot(3, 3, 7 + idx)
        plt.title(f'{method_name} Kenar Tespiti')
        plt.imshow(edges, cmap='gray')
        plt.axis('off')
        
        # Kontur tespiti
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > largest_area:
                largest_area = area
                best_contour = contour
                best_method_name = method_name
                shape, vertices = detect_shape(contour)
                if shape != 'Belirsiz':
                    best_shape, best_vertices = shape, vertices
    
    # En iyi kontur ve şekil bilgisi
    if best_contour is not None:
        img_with_contours = image.copy()
        cv2.drawContours(img_with_contours, [best_contour], -1, (0,255,0), 3)
        plt.subplot(3, 3, 9)
        plt.title(f'En İyi Kontur\nŞekil: {best_shape}\nKöşe Sayısı: {best_vertices}\nYöntem: {best_method_name}')
        plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    else:
        plt.subplot(3, 3, 9)
        plt.title('Kontur Bulunamadı')
        plt.axis('off')

    plt.tight_layout()
    plt.suptitle(f'İşlem Adımları - {filename}', fontsize=16)
    plt.show()

    return best_contour, best_shape

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
    
    num_vertices = len(approx)
    
    if num_vertices == 3:
        # Üçgen için ek kontrol
        angles = []
        for i in range(3):
            pt1 = approx[i][0]
            pt2 = approx[(i+1) % 3][0]
            pt3 = approx[(i+2) % 3][0]
            angle = abs(math.degrees(math.atan2(pt3[1]-pt2[1], pt3[0]-pt2[0]) - math.atan2(pt1[1]-pt2[1], pt1[0]-pt2[0])))
            angles.append(angle)
        
        # Üçgenin açılarını kontrol et
        if all(30 < angle < 150 for angle in angles):
            return 'Üçgen', num_vertices
    
    elif num_vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        return 'Kare' if 0.9 <= aspect_ratio <= 1.1 else 'Dikdörtgen', num_vertices
    elif num_vertices == 8:
        return 'Sekizgen', num_vertices
    elif num_vertices > 8:
        return 'Daire', num_vertices
    else:
        return 'Bilinmeyen', num_vertices

def classify_traffic_symbols(input_folder, output_folder):
    # Çıktı klasörünü temizle
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
    # Şekil klasörlerini oluştur
    shapes = ['Üçgen', 'Daire', 'Kare', 'Dikdörtgen', 'Sekizgen', 'Bilinmeyen']
    for shape in shapes:
        os.makedirs(os.path.join(output_folder, shape), exist_ok=True)
    
    # Tüm resimleri işle
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue
        
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Resim okunamadı: {filename}")
            continue
        
        # Görüntü işleme ve görselleştirme
        best_contour, best_shape = process_and_visualize(image, filename)
        
        output_path = os.path.join(output_folder, best_shape, filename)
        shutil.copy(image_path, output_path)
        print(f"İşlenen resim: {filename}, Tespit edilen şekil: {best_shape}")
    
    print("Sınıflandırma tamamlandı!")

# Kullanım örneği
input_folder = 'traffic_Data/DATA/mix'  # Giriş klasörü
output_folder = 'classified_symbols'  # Çıkış klasörü

classify_traffic_symbols(input_folder, output_folder)