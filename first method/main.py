import cv2
import numpy as np
import os
import shutil

def classify_traffic_symbols(input_folder, output_folder):
    # Çıktı klasörlerini oluştur
    shapes = ['triangle', 'circle', 'rectangle', 'octagon']
    for shape in shapes:
        os.makedirs(os.path.join(output_folder, shape), exist_ok=True)
    
    # Tüm resimleri işle
    for filename in os.listdir(input_folder):
        # Resmi oku
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        # Griye çevir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Kenarları tespit et
        edges = cv2.Canny(gray, 50, 150)
        
        # Konturları bul
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Konturun yaklaşık şeklini hesapla
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            
            # Şekli belirle
            if len(approx) == 3:
                shape = 'triangle'
            elif len(approx) == 4:
                # Kare mi, dikdörtgen mi kontrolü
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                shape = 'rectangle' if 0.9 <= aspect_ratio <= 1.1 else 'rectangle'
            elif len(approx) == 8:
                shape = 'octagon'
            elif len(approx) > 8:
                shape = 'circle'
            else:
                continue  # Tanımlanamayan şekiller
            
            # Resmi ilgili klasöre kopyala
            output_path = os.path.join(output_folder, shape, filename)
            shutil.copy(image_path, output_path)
    
    print("Sınıflandırma tamamlandı!")

# Kullanım örneği
input_folder = 'traffic_Data\\DATA\\0'  # Giriş klasörü
output_folder = 'classified_symbols'  # Çıkış klasörü

classify_traffic_symbols(input_folder, output_folder)