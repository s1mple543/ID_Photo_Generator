import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        pass
    
    def resize(self, image, width=None, height=None):
        h, w = image.shape[:2]
        if width is None and height is None:
            return image
        
        if width is None:
            ratio = height / float(h)
            dim = (int(w * ratio), height)
        else:
            ratio = width / float(w)
            dim = (width, int(h * ratio))
            
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    
    def rotate(self, image, angle):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    
    def apply_filter(self, image, filter_type):
        if filter_type == "grayscale":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif filter_type == "sepia":
            kernel = np.array([[0.272, 0.534, 0.131],
                              [0.349, 0.686, 0.168],
                              [0.393, 0.769, 0.189]])
            return cv2.transform(image, kernel)
        elif filter_type == "blur":
            return cv2.GaussianBlur(image, (15, 15), 0)
        elif filter_type == "edge":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.Canny(gray, 100, 200)
        return image