# import os
# import re

# import cv2
# import easyocr

# # Path to images and output file
# IMAGE_DIR = "output/images"
# OUTPUT_FILE = "output/extracted_code.py"

# # Create an EasyOCR reader
# reader = easyocr.Reader(["en"])  # Specify the language(s) you want to use


# def clean_extracted_text(text):
#     """
#     Cleans up extracted text by fixing common OCR mistakes.
#     """
#     text = re.sub(
#         r"(\d+)\s+(\w)", r"\1 \2", text
#     )  # Fix spacing between numbers & words
#     text = re.sub(r"([a-zA-Z0-9])([.,;:])", r"\1 \2", text)  # Space before punctuation
#     text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with single space
#     return text.strip()


# # Process each image and extract Django code
# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     for image_file in os.listdir(IMAGE_DIR):
#         image_path = os.path.join(IMAGE_DIR, image_file)
#         img = cv2.imread(image_path)

#         print(f"🔍 Checking Image: {image_file}...")

#         if img is None:
#             print(
#                 f"❌ Error: Could not load {image_file}. Check if it's a valid image file."
#             )
#             continue  # Skip this file

#         print(f"🔍 Extracting text from: {image_file}...")

#         # Convert to grayscale
#         gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         print(f"🔍 Converting Image to grayscale: {image_file}...")

#         # Resize image (helps with small text)
#         scale_factor = 2  # Increase size for better OCR accuracy
#         resized_image = cv2.resize(
#             gray_image,
#             None,
#             fx=scale_factor,
#             fy=scale_factor,
#             interpolation=cv2.INTER_CUBIC,
#         )
#         print(f"🔍 Resizing image for better OCR accuracy: {image_file}...")

#         # Apply adaptive thresholding
#         thresh_image = cv2.adaptiveThreshold(
#             resized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
#         )
#         print(f"🔍 Applying adaptive thresholding: {image_file}...")

#         # Extract text using EasyOCR
#         result = reader.readtext(thresh_image)
#         print(f"🔍 Extracting text using EasyOCR: {image_file}...")

#         # Debug: Print raw OCR results before cleaning
#         extracted_text = "\n".join([text[1] for text in result])

#         # Debug: Log before saving
#         print(
#             f"📜 Extracted text before cleaning from {image_file}:\n{extracted_text}\n"
#         )

#         if not extracted_text.strip():
#             print(
#                 f"❌ No text detected in {image_file}. Check the image quality or preprocessing."
#             )
#             continue  # Skip saving if no text was detected

#         # Clean up the extracted text
#         cleaned_text = clean_extracted_text(extracted_text)

#         # Debug: Log before saving
#         print(f"📜 Extracted text from {image_file}:\n{cleaned_text}\n")

#         # Save formatted Django code
#         f.write(f"# Code extracted from {image_file}\n")
#         f.write(cleaned_text + "\n\n")

# print(f"✅ Django code extracted and saved to {OUTPUT_FILE}")


import os
import re

import cv2
import easyocr
import numpy as np

# Path to images and output file
IMAGE_DIR = "output/debug"
OUTPUT_FILE = "output/extracted_code.py"

# Create an EasyOCR reader with GPU disabled (force CPU mode)
reader = easyocr.Reader(["en"], gpu=False)  # Ensure CPU compatibility


def clean_extracted_text(text):
    """
    Cleans up extracted text by fixing common OCR mistakes.
    """
    text = re.sub(
        r"(\d+)\s+(\w)", r"\1 \2", text
    )  # Fix spacing between numbers & words
    text = re.sub(r"([a-zA-Z0-9])([.,;:])", r"\1 \2", text)  # Space before punctuation
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with single space
    return text.strip()


def preprocess_image(image, image_file):
    """
    Enhances image contrast and reduces noise to improve OCR accuracy.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray)

    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(enhanced_image, (3, 3), 0)

    # Apply adaptive thresholding
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Sharpening filter
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(thresholded, -1, kernel)

    # Dilation (thickens text for better OCR recognition)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(sharpened, kernel, iterations=1)

    # Save debug images
    debug_dir = "output/debug"
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(f"{debug_dir}/{image_file}_gray.jpg", gray)
    cv2.imwrite(f"{debug_dir}/{image_file}_enhanced.jpg", enhanced_image)
    cv2.imwrite(f"{debug_dir}/{image_file}_thresholded.jpg", thresholded)
    cv2.imwrite(f"{debug_dir}/{image_file}_sharpened.jpg", sharpened)
    cv2.imwrite(f"{debug_dir}/{image_file}_dilated.jpg", dilated)

    return gray, enhanced_image, thresholded, sharpened, dilated


# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Process each image and extract Django code
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for image_file in os.listdir(IMAGE_DIR):
        image_path = os.path.join(IMAGE_DIR, image_file)
        img = cv2.imread(image_path)

        print(f"🔍 Checking Image: {image_file}...")

        if img is None:
            print(
                f"❌ Error: Could not load {image_file}. Check if it's a valid image file."
            )
            continue  # Skip this file

        print(f"🔍 Enhancing Image for OCR: {image_file}...")

        # Preprocess image with multiple methods
        gray, enhanced, thresholded, sharpened, dilated = preprocess_image(
            img, image_file
        )

        # Try EasyOCR on multiple versions of the image
        ocr_results = []
        for processed_img in [gray, enhanced, thresholded, sharpened, dilated]:
            result = reader.readtext(processed_img)
            if result:
                ocr_results = result
                break  # Stop if OCR finds text

        if not ocr_results:
            print(
                f"❌ No text detected in {image_file}. Check the debug images for preprocessing issues."
            )
            continue  # Skip saving if no text was detected

        # Extract text from result
        extracted_text = "\n".join([text[1] for text in ocr_results])

        # Debug: Print extracted text
        print(f"📜 Extracted text from {image_file}:\n{extracted_text}\n")

        # Clean up the extracted text
        cleaned_text = clean_extracted_text(extracted_text)

        # Save extracted text to file
        f.write(f"# Code extracted from {image_file}\n")
        f.write(cleaned_text + "\n\n")

print(f"✅ Django code extracted and saved to {OUTPUT_FILE}")
