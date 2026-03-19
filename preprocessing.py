import cv2
import numpy as np

def preprocess_image(image_bytes):
    # 1. Convert the raw memory bytes into a NumPy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # 2. Decode the array into an OpenCV image directly in memory
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Could not decode the uploaded image.")

    # 3. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 4. Noise removal
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 5. Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    return thresh