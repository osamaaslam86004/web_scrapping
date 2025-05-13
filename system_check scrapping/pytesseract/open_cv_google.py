import cv2
import pytesseract

# Load the image
image = cv2.imread('path_to_image.png')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

# Use Tesseract to do OCR on the processed image
extracted_text = pytesseract.image_to_string(thresh_image)

print(extracted_text)