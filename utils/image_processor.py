import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class ImageProcessor:
    def __init__(self):
        pass
    
    def rotate(self, image, angle):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    
    def translate(self, image, x, y):
        h, w = image.shape[:2]
        M = np.float32([[1, 0, x], [0, 1, y]])
        return cv2.warpAffine(image, M, (w, h))
    
    def apply_mirror(self, image, mirror_type):
        if mirror_type == "horizontal_mirroring":
            return cv2.flip(image, 1)
        elif mirror_type == "vertical_mirroring":
            return cv2.flip(image, 0)
        return image
    
    def apply_affine_transform(self, image):
        h, w = image.shape[:2]
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        M = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(image, M, (w, h))
    
    def apply_space(self, image, operation):
        if operation == "histogram_drawing":
            if len(image.shape) == 2:  # Grayscale
                hist = cv2.calcHist([image], [0], None, [256], [0, 256])
                fig = plt.figure(figsize=(4, 4))
                plt.plot(hist)
                plt.title('Grayscale Histogram')
                plt.xlabel('Pixel Value')
                plt.ylabel('Frequency')
            else:  # Color
                fig = plt.figure(figsize=(8, 4))
                colors = ('b', 'g', 'r')
                for i, col in enumerate(colors):
                    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                    plt.plot(hist, color=col)
                    plt.xlim([0, 256])
                plt.title('Color Histogram')
                plt.xlabel('Pixel Value')
                plt.ylabel('Frequency')
            
            canvas = FigureCanvas(fig)
            canvas.draw()
            img = np.array(canvas.renderer.buffer_rgba())
            plt.close(fig)
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
        elif operation == "histogram_equalization":
            if len(image.shape) == 2:  # Grayscale
                return cv2.equalizeHist(image)
            else:  # Color (apply to Y channel in YCrCb)
                ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
                return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
                
        elif operation == "grayscale_transformation":
            # Logarithmic transformation
            c = 255 / np.log(1 + np.max(image))
            log_transformed = c * np.log(1 + image.astype(np.float32))
            return np.uint8(log_transformed)
            
        elif operation == "smoothing_filtering":
            return cv2.GaussianBlur(image, (5, 5), 0)
            
        elif operation == "sharpen_filtering":
            kernel = np.array([[-1, -1, -1], 
                              [-1, 9, -1], 
                              [-1, -1, -1]])
            return cv2.filter2D(image, -1, kernel)
            
        return image
    
    def apply_frequency(self, image, operation):
        if len(image.shape) > 2:  # Convert to grayscale if color
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Fourier Transform
        dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # Create filter
        rows, cols = image.shape
        crow, ccol = rows//2, cols//2
        mask = np.zeros((rows, cols, 2), np.uint8)
        
        if operation == "lowpass_filtering":
            r = 30  # Radius
            cv2.circle(mask, (ccol, crow), r, (1, 1), -1)
        elif operation == "highpass_filtering":
            r = 30  # Radius
            cv2.circle(mask, (ccol, crow), r, (1, 1), -1)
            mask = 1 - mask
        elif operation == "bandpass_filtering":
            r_out = 60
            r_in = 20
            cv2.circle(mask, (ccol, crow), r_out, (1, 1), -1)
            cv2.circle(mask, (ccol, crow), r_in, (0, 0), -1)
        else:
            return image
            
        # Apply filter and inverse DFT
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        
        # Normalize
        cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
        return img_back.astype(np.uint8)
    
    def apply_Morphological(self, image, operation):
        kernel = np.ones((5,5), np.uint8)
        if operation == "open_operation":
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == "close_operation":
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return image
    
    def apply_edge_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)