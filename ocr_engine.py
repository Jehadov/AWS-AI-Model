import pytesseract
from PIL import Image
import platform

# Only use the Windows path if running locally on Windows. 
# In Docker (Linux), it defaults to the system path.
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def run_ocr(image, language="eng"):
    text = pytesseract.image_to_string(
        Image.fromarray(image),
        lang=language
    )
    return text