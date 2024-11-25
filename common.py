import cv2

def detect_shape(contour):
    """Advanced function to detect the shape of a contour"""
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    num_vertices = len(approx)
    
    if num_vertices == 3:
        return 'triangle', num_vertices
    elif num_vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        return 'rectangle' if 0.9 <= aspect_ratio <= 1.1 else 'rectangle', num_vertices
    elif num_vertices == 5:
        return 'pentagon', num_vertices
    elif num_vertices == 8:
        return 'octagon', num_vertices
    elif num_vertices > 8:
        return 'circle', num_vertices
    else:
        return 'unknown', num_vertices