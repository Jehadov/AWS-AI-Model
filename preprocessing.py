import cv2
import numpy as np

def preprocess_image(image_path):
    #Load image
    image = cv2.imread(image_path)
    #Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Noise removal
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    #Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresh