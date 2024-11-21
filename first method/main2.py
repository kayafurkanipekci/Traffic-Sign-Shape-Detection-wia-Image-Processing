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

def classify_traffic_symbols(input_folder, output_folder):
    # Çıktı klasörünü temizle
    clean_output_folder(output_folder)
    
    # Şekil klasörlerini oluştur
    shapes = ['triangle', 'circle', 'rectangle', 'octagon']
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
        
        # Görüntüyü ön işleme
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Bulanıklaştırma (Gürültüyü azaltmak için)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Eşikleme (Threshold)
        # Otomatik eşikleme (Adaptive Thresholding)
        thresh = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )
        
        # Kenarları tespit et
        edges = cv2.Canny(thresh, 50, 200)
        
        # Konturları bul
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # En büyük kontur (ana şekil) için alan kontrolü
        main_contour = max(contours, key=cv2.contourArea)
        
        # Konturun yaklaşık şeklini hesapla
        perimeter = cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, 0.04 * perimeter, True)
        
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
        else:
            shape = 'circle'
        
        # Resmi ilgili klasöre kopyala
        output_path = os.path.join(output_folder, shape, filename)
        shutil.copy(image_path, output_path)
        
        # Görselleştirme (opsiyonel)
        plt.figure(figsize=(15,5))
        
        plt.subplot(151)
        plt.title('Orijinal')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        plt.subplot(152)
        plt.title('Gri')
        plt.imshow(gray, cmap='gray')
        
        plt.subplot(153)
        plt.title('Bulanık')
        plt.imshow(blurred, cmap='gray')
        
        plt.subplot(154)
        plt.title('Eşikleme')
        plt.imshow(thresh, cmap='gray')
        
        plt.subplot(155)
        plt.title('Kenarlar')
        plt.imshow(edges, cmap='gray')
        
        plt.suptitle(f'Şekil: {shape}')
        plt.tight_layout()
        plt.show()
    
    print("Sınıflandırma tamamlandı!")

# Kullanım örneği
input_folder = 'traffic_Data\DATA\mix'  # Giriş klasörü
output_folder = 'classified_symbols'  # Çıkış klasörü

classify_traffic_symbols(input_folder, output_folder)