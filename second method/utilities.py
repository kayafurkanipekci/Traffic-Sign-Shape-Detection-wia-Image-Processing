from cv2 import arcLength, approxPolyDP

def getCornerPoints(cont):
    peri = arcLength(cont, True)
    approx = approxPolyDP(cont, 0.02 * peri, True)
    return approx