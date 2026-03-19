import platform
import pytesseract

# THE FIX: You must import the preprocessing function from your other file!
from preprocessing import preprocess_image

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def run_ocr(image_bytes, language="eng+ara"):
    # 1. Clean the image using your updated OpenCV function
    cleaned_image = preprocess_image(image_bytes)
    
    # 2. Pass the cleaned OpenCV image directly into Tesseract
    text = pytesseract.image_to_string(
        cleaned_image,
        lang=language
    )
    
    return text