import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

def clean_output_folder(output_folder):
    """Önceki sınıflandırma çıktılarını temizler"""
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

def detect_shape(contour):
    """Konturun şeklini tespit eden gelişmiş fonksiyon"""
    # Konturun yaklaşık şeklini hesapla
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    
    # Şekli belirle
    num_vertices = len(approx)
    
    if num_vertices == 3:
        return 'triangle', num_vertices
    elif num_vertices == 4:
        # Kare mi, dikdörtgen mi kontrolü
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        return 'rectangle' if 0.9 <= aspect_ratio <= 1.1 else 'rectangle', num_vertices
    elif num_vertices == 8:
        return 'octagon', num_vertices
    elif num_vertices > 8:
        return 'circle', num_vertices
    else:
        return 'unknown', num_vertices

def classify_traffic_symbols(input_folder, output_folder):
    # Çıktı klasörünü temizle
    clean_output_folder(output_folder)
    
    # Şekil klasörlerini oluştur
    shapes = ['triangle', 'circle', 'rectangle', 'octagon', 'unknown']
    for shape in shapes:
        os.makedirs(os.path.join(output_folder, shape), exist_ok=True)
    
    # Tüm resimleri işle
    for filename in os.listdir(input_folder):
        # Yalnızca resim dosyalarını işle
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue
        
        # Resmi oku
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        # Resim okunamazsa atla
        if image is None:
            print(f"Resim okunamadı: {filename}")
            continue
        
        # Görüntüyü önişleme
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Bulanıklaştırma (Gürültüyü azaltmak için)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Çoklu eşikleme teknikleri
        methods = [
            ('Otsu Binary', cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
            ('Adaptive Gaussian', cv2.adaptiveThreshold(blurred, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)),
            ('Adaptive Mean', cv2.adaptiveThreshold(blurred, 255, 
                                cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2))
        ]
        
        # En büyük alan ve şekli saklamak için değişkenler
        largest_area = 0
        best_shape = None
        best_vertices = 0
        best_method_name = None
        best_contour = None
        
        # Her yöntem için kontrol
        for method_name, thresh in methods:
            # Kenarları tespit et
            edges = cv2.Canny(thresh, 50, 200)
            
            # Konturları bul
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Kontür alan filtresi
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
            
            if contours:
                # Her kontür için şekil tespiti
                for contour in contours:
                    # Konturun yaklaşık şeklini hesapla
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.09 * perimeter, True)
                    
                    # Alan ve şekil bilgisi
                    area = cv2.contourArea(contour)
                    shape, vertices = detect_shape(contour)
                    
                    # En büyük alanı ve ilgili bilgileri güncelle
                    if area > largest_area:
                        largest_area = area
                        best_shape = shape
                        best_vertices = vertices
                        best_method_name = method_name
                        best_contour = contour
        
        # Eğer hiçbir yöntemle kontür bulunamadıysa atla
        if best_contour is None:
            print(f"Kontür bulunamadı: {filename}")
            continue
        
        # Resmi ilgili klasöre kopyala
        output_path = os.path.join(output_folder, best_shape, filename)
        shutil.copy(image_path, output_path)
        
        # Görselleştirme
        plt.figure(figsize=(20,6))
        
        # Orijinal görüntü ve kontür çizilmiş görüntü
        plt.subplot(1, 2, 1)
        plt.title('Orijinal')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # Kontürler çizilmiş görüntü
        img_with_contours = image.copy()
        cv2.drawContours(img_with_contours, [best_contour], -1, (0,255,0), 3)
        
        plt.subplot(1, 2, 2)
        plt.title(f'Şekil: {best_shape} (Köşe Sayısı: {best_vertices})\nYöntem: {best_method_name}')
        plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print("Sınıflandırma tamamlandı!")

# Kullanım örneği
input_folder = 'traffic_Data\DATA\mix'  # Giriş klasörü
output_folder = 'classified_symbols'  # Çıkış klasörü

classify_traffic_symbols(input_folder, output_folder)