import cv2
import numpy as np
import json

class GroundTruthLabeler:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.original_image = self.image.copy()
        self.points = []
        self.completed = False

    def mouse_callback(self, event, x, y, flags, param):
        img_copy = self.original_image.copy()
        
        # Mevcut noktaları çiz
        if len(self.points) > 0:
            for i in range(len(self.points)):
                cv2.circle(img_copy, self.points[i], 5, (0, 255, 0), -1)
                
                # Noktaları birleştir
                if i > 0:
                    cv2.line(img_copy, self.points[i-1], self.points[i], (0, 255, 0), 2)
        
        # İlk ve son noktayı birleştir (polygon kapatma)
        if len(self.points) > 2:
            cv2.line(img_copy, self.points[0], self.points[-1], (0, 255, 0), 2)

        # Sol tıklama: nokta ekleme
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))

        # Sağ tıklama: son noktayı silme
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()

        # Çizimi güncelle
        cv2.imshow('Image', img_copy)

    def label(self):
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.mouse_callback)

        print("Trafik sembolünün köşelerini tıklayın:")
        print("Sol tık: Nokta ekle")
        print("Sağ tık: Son noktayı sil")
        print("Enter: Etiketlemeyi tamamla")
        print("ESC: İptal et")

        while True:
            cv2.imshow('Image', self.original_image)
            key = cv2.waitKey(1) & 0xFF

            # Enter tuşu: Etiketlemeyi tamamla
            if key == 13:
                if len(self.points) >= 3:
                    shape = input("Şekli girin (circle, triangle, rectangle, octagon): ")
                    
                    # Kontur bilgilerini kaydet
                    ground_truth = {
                        'shape': shape,
                        'contour': self.points
                    }
                    
                    # JSON'a kaydet
                    with open('ground_truth.json', 'w') as f:
                        json.dump(ground_truth, f, default=lambda x: list(map(int, x)))
                    
                    cv2.destroyAllWindows()
                    return ground_truth
                else:
                    print("En az 3 nokta gerekli!")

            # ESC tuşu: İptal et
            elif key == 27:
                cv2.destroyAllWindows()
                return None

# Kullanım
labeler = GroundTruthLabeler('traffic_Data\\DATA\\mix\\038_1_0006.png')
ground_truth = labeler.label()
print(ground_truth)
