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
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    
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
        
        # Görüntüyü ön işleme
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Bulanıklaştırma (Gürültüyü azaltmak için)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Çoklu eşikleme teknikleri denemek için
        methods = [
            ('Binary', cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
            ('Adaptive Gaussian', cv2.adaptiveThreshold(blurred, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)),
            ('Adaptive Mean', cv2.adaptiveThreshold(blurred, 255, 
                                cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2))
        ]
        
        # En iyi sonucu bulmak için
        best_contours = None
        best_method = None
        
        for method_name, thresh in methods:
            # Kenarları tespit et
            edges = cv2.Canny(thresh, 50, 200)
            
            # Konturları bul
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Kontür alan filtresi
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
            
            if contours:
                # En büyük konturu bul
                main_contour = max(contours, key=cv2.contourArea)
                
                # Şekil tespiti
                shape, vertices = detect_shape(main_contour)
                
                # Kontür bulundu ve şekil tespit edildi ise
                best_contours = contours
                best_method = method_name
                break
        
        # Eğer hiçbir yöntemle kontür bulunamadıysa atla
        if best_contours is None:
            print(f"Kontür bulunamadı: {filename}")
            continue
        
        # Şekli belirle
        shape, vertices = detect_shape(max(best_contours, key=cv2.contourArea))
        
        # Resmi ilgili klasöre kopyala
        output_path = os.path.join(output_folder, shape, filename)
        shutil.copy(image_path, output_path)
        
        # Görselleştirme
        plt.figure(figsize=(20,6))
        
        # Orijinal görüntü
        plt.subplot(161)
        plt.title('Orijinal')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # Gri tonlama
        plt.subplot(162)
        plt.title('Gri')
        plt.imshow(gray, cmap='gray')
        plt.axis('off')
        
        # Bulanıklaştırılmış görüntü
        plt.subplot(163)
        plt.title('Bulanık')
        plt.imshow(blurred, cmap='gray')
        plt.axis('off')
        
        # Eşikleme yöntemleri
        for i, (method_name, thresh) in enumerate(methods, start=4):
            plt.subplot(1, 6, i)
            plt.title(method_name)
            plt.imshow(thresh, cmap='gray')
            plt.axis('off')
        
        # Kontürler çizilmiş görüntü
        plt.figure(figsize=(10,6))
        plt.title(f'Şekil: {shape} (Köşe Sayısı: {vertices})')
        
        # Kontürü çizilen görüntü
        img_with_contours = image.copy()
        cv2.drawContours(img_with_contours, best_contours, -1, (0,255,0), 3)
        plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print("Sınıflandırma tamamlandı!")

# Kullanım örneği
input_folder = 'traffic_Data\DATA\mix'  # Giriş klasörü
output_folder = 'classified_symbols'  # Çıkış klasörü

classify_traffic_symbols(input_folder, output_folder)