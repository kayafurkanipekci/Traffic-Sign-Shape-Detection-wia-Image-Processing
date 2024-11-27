import json
import cv2
import numpy as np

class GroundTruthEvaluator:
    def __init__(self, ground_truth_file, predicted_results):
        """
        ground_truth_file: JSON formatında manuel etiketlenmiş görüntü bilgileri
        predicted_results: Algoritmanızın tahmin sonuçları
        """
        with open(ground_truth_file, 'r') as f:
            self.ground_truth = json.load(f)
        
        self.predicted_results = predicted_results
    
    def calculate_iou(self, ground_truth_box, predicted_box):
        """
        Intersection over Union (IoU) hesaplama
        Nesnenin konumsal doğruluğunu değerlendirir
        """
        # Kesişim alanını hesapla
        x1 = max(ground_truth_box[0], predicted_box[0])
        y1 = max(ground_truth_box[1], predicted_box[1])
        x2 = min(ground_truth_box[0] + ground_truth_box[2], 
                 predicted_box[0] + predicted_box[2])
        y2 = min(ground_truth_box[1] + ground_truth_box[3], 
                 predicted_box[1] + predicted_box[3])
        
        # Kesişim alanı
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Birleşim alanı
        ground_truth_area = ground_truth_box[2] * ground_truth_box[3]
        predicted_area = predicted_box[2] * predicted_box[3]
        union_area = ground_truth_area + predicted_area - intersection_area
        
        # IoU hesaplama
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou
    
    def evaluate_detection(self, iou_threshold=0.5, confidence_threshold=0.7):
        """
        Nesne tespiti performansını değerlendirir
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for image_name, ground_truth_data in self.ground_truth.items():
            # Tahmin sonuçları içinden ilgili görüntüyü bul
            predicted_data = self.predicted_results.get(image_name, [])
            
            for gt_object in ground_truth_data['objects']:
                matched = False
                
                for pred_object in predicted_data:
                    # Şekil ve sınıf kontrolü
                    if (gt_object['shape'] == pred_object['shape'] and 
                        pred_object['confidence'] >= confidence_threshold):
                        
                        # IoU hesaplama
                        iou = self.calculate_iou(
                            gt_object['bbox'], 
                            pred_object['bbox']
                        )
                        
                        # Eşik değerinin üzerindeyse doğru tespit
                        if iou >= iou_threshold:
                            true_positives += 1
                            matched = True
                            break
                
                # Eşleşme yoksa yanlış negatif
                if not matched:
                    false_negatives += 1
        
        # Yanlış pozitifler
        false_positives = len(self.predicted_results) - true_positives
        
        # Performans metrikleri
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

# Örnek kullanım
ground_truth_data = {
    'image1.jpg': {
        'objects': [
            {
                'shape': 'triangle',
                'bbox': [100, 100, 50, 50]  # x, y, width, height
            }
        ]
    }
}

predicted_results = {
    'image1.jpg': [
        {
            'shape': 'triangle',
            'confidence': 0.85,
            'bbox': [105, 105, 45, 48]
        }
    ]
}

evaluator = GroundTruthEvaluator(ground_truth_data, predicted_results)
performance = evaluator.evaluate_detection()
print(performance)