# import os
# import re

# import cv2
# import numpy as np
# import pytesseract

# # Path to images and output file
# IMAGE_DIR = "output/images"
# OUTPUT_FILE = "output/extracted_code.py"

# # Configure Tesseract path (Only needed for Windows)
# TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH  # Remove this on
# Linux/macOS

# # Tesseract OCR configuration to improve text extraction
# # TESSERACT_CONFIG = "--psm 4 -c preserve_interword_spaces=1 tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789():,_=.\"' \n\t"

# TESSERACT_CONFIG = "--oem 3 --psm 4 -c preserve_interword_spaces=1"


# def clean_extracted_text(text):
#     text = text.replace(" :", ":").replace(" ,", ",")  # Fix misplaced punctuation
#     text = re.sub(r"(\w)([A-Z])", r"\1 \2", text)  # Add space between camel-case words
#     text = re.sub(r"(\d+)(\w)", r"\1 \2", text)  # Fix number-letter merging
#     text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
#     # Remove standalone numbers (likely from line numbers)
#     text = re.sub(r"\b\d{1,2}\b", "", text)
#     text = text.replace("|", ":")

#     return text.strip()


# # Process each image and extract Django code
# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     for image_file in os.listdir(IMAGE_DIR):
#         image_path = os.path.join(IMAGE_DIR, image_file)
#         img = cv2.imread(image_path)

#         print(f"🔍 Extracting text from: {image_file}...")

#         # Convert to grayscale
#         gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Apply Gaussian blur to reduce noise
#         # blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
#         denoised_image = cv2.bilateralFilter(gray_image, 9, 75, 75)

#         # Apply adaptive thresholding
#         # thresh_image = cv2.adaptiveThreshold(
#         #     blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
#         # )

#         # thresh_image = cv2.adaptiveThreshold(
#         #     denoised_image,
#         #     255,
#         #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         #     cv2.THRESH_BINARY,
#         #     11,
#         #     2,
#         # )

#         _, otsu_thresh = cv2.threshold(
#             denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
#         )

#         thresh_image = otsu_thresh

#         kernel = np.ones((1, 1), np.uint8)
#         morph_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)

#         # Get bounding box estimates
#         print(pytesseract.image_to_boxes(thresh_image))

#         # Get verbose data including boxes, confidences, line and page numbers
#         print(pytesseract.image_to_data(thresh_image))

#         # Get information about orientation and script detection
#         print(pytesseract.image_to_osd(thresh_image))

#         # Extract text using OCR
#         extracted_text = pytesseract.image_to_string(
#             thresh_image, lang="eng", config=TESSERACT_CONFIG
#         )

#         # Clean up the extracted text
#         cleaned_text = clean_extracted_text(extracted_text)

#         # Save formatted Django code
#         if cleaned_text:  # Only save if there's valid code
#             f.write(f"# Code extracted from {image_file}\n")
#             f.write(cleaned_text + "\n\n")

# print(f"✅ Django code extracted and saved to {OUTPUT_FILE}")


# import os
# import re

# import cv2
# import numpy as np
# import pytesseract

# # Path to images and output file
# IMAGE_DIR = "output/images"
# OUTPUT_FILE = "output/extracted_code.py"

# # Configure Tesseract path (Only needed for Windows)
# TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH  # Remove this on
# Linux/macOS

# # Improved Tesseract OCR configuration
# TESSERACT_CONFIG = "--oem 3 --psm 4 -c preserve_interword_spaces=1"


# def clean_extracted_text(text):
#     """Cleans and formats extracted text to match proper Python/Django syntax."""
#     # Fix misplaced punctuation
#     text = text.replace(" :", ":").replace(" ,", ",")

#     # Fix broken spacing between words and symbols
#     text = re.sub(r"(\w)([A-Z])", r"\1 \2", text)  # Add space in camelCase words
#     text = re.sub(r"(\d+)(\w)", r"\1 \2", text)  # Fix number-letter merging
#     text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
#     text = text.replace("|", ":")  # Fix common OCR mistake
#     text = text.replace(") :", "):")  # Fix method declaration issues
#     text = text.replace("(", " (").replace(")", ") ")
#     # Remove unwanted characters
#     text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII characters
#     text = re.sub(r"\b[A-Z]\b", "", text)  # Remove standalone uppercase letters
#     text = re.sub(r"(\w)(#)", r"\1 \2", text)  # add space before #
#     text = text.replace("  ", " ")  # remove double space
# text = re.sub(r"\b\d{1,2}\s", "", text)  # Remove standalone numbers and
# next space

#     # split in line
#     lines = text.split("\n")
#     new_lines = []
#     indentation_level = 0
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue

#         # remove some word that might generate from the image
#         line = line.replace("b ", "").replace("h ", "")
#         if line.startswith(("def ", "class ")):
#             indentation_level = 0
#             new_lines.append(line)
#             if ":" in line:
#                 indentation_level += 1

#         elif ":" in line:
#             new_lines.append(("    " * indentation_level) + line)
#             indentation_level += 1

#         elif line.startswith(("return ")):
#             indentation_level -= 1 if indentation_level > 0 else 0
#             new_lines.append("    " * indentation_level + line)

#         elif any(
#             keyword in line
#             for keyword in [
#                 "for ",
#                 "if ",
#                 "else:",
#                 "elif ",
#                 "while ",
#                 "try:",
#                 "except ",
#             ]
#         ):
#             new_lines.append("    " * indentation_level + line)
#             indentation_level += 1

#         else:
#             new_lines.append(("    " * indentation_level) + line)

#     return "\n".join(new_lines)


# def preprocess_image(image_path):
#     """Preprocesses an image to improve OCR accuracy."""
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Error: Could not load image at {image_path}")
#         return None

#     # Convert to grayscale
#     gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Apply noise reduction
#     denoised_image = cv2.bilateralFilter(gray_image, 9, 75, 75)

#     # Apply Otsu's thresholding
#     _, otsu_thresh = cv2.threshold(
#         denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
#     )

#     return otsu_thresh


# # Process each image and extract Django code
# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     for image_file in os.listdir(IMAGE_DIR):
#         image_path = os.path.join(IMAGE_DIR, image_file)

#         print(f"🔍 Extracting text from: {image_file}...")

#         processed_image = preprocess_image(image_path)
#         if processed_image is None:
#             continue  # Skip this file if it couldn't be loaded

#         # Extract text using OCR
#         extracted_text = pytesseract.image_to_string(
#             processed_image, lang="eng", config=TESSERACT_CONFIG
#         )

#         # Clean and format the extracted text
#         cleaned_text = clean_extracted_text(extracted_text)

#         # Save formatted Django code
#         if cleaned_text:  # Only save if there's valid code
#             f.write(f"# Code extracted from {image_file}\n")
#             f.write(cleaned_text + "\n\n")

# print(f"✅ Django code extracted and saved to {OUTPUT_FILE}")


# import os
# import re

# import cv2
# import numpy as np
# import pytesseract

# # Path to images and output file
# IMAGE_DIR = "output/images"
# OUTPUT_FILE = "output/extracted_code.py"

# # Configure Tesseract path (Only needed for Windows)
# TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH  # Remove this on
# Linux/macOS

# # Improved Tesseract OCR configuration
# TESSERACT_CONFIG = "--oem 3 --psm 4 -c preserve_interword_spaces=1"
# import os
# import re

# import cv2
# import numpy as np
# import pytesseract
# from format_python_code import format_code

# # Path to images and output file
# IMAGE_DIR = "output/images"
# OUTPUT_FILE = "output/extracted_code.py"

# # Configure Tesseract path (Only needed for Windows)
# TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH  # Remove this on
# Linux/macOS

# # Improved Tesseract OCR configuration
# TESSERACT_CONFIG = "--oem 3 --psm 4 -c preserve_interword_spaces=1"


# def clean_extracted_text(text):
#     """Cleans the extracted text to prepare it for the formatter."""
#     # Fix misplaced punctuation
#     text = text.replace(" :", ":").replace(" ,", ",")

#     # Fix broken spacing between words and symbols
#     text = re.sub(r"(\w)([A-Z])", r"\1 \2", text)  # Add space in camelCase words
#     text = re.sub(r"(\d+)(\w)", r"\1 \2", text)  # Fix number-letter merging
#     text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
#     text = text.replace("|", ":")  # Fix common OCR mistake
#     text = text.replace(") :", "):")  # Fix method declaration issues
#     text = text.replace("(", " (").replace(")", ") ")
#     # Remove unwanted characters
#     text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII characters
#     text = re.sub(r"\b[A-Z]\b", "", text)  # Remove standalone uppercase letters
#     text = re.sub(r"(\w)(#)", r"\1 \2", text)  # add space before #
#     text = text.replace("  ", " ")  # remove double space
#     text = re.sub(r"\b\d{1,2}\s", "", text)  # Remove standalone numbers and next space
#     text = text.replace("b ", "").replace("h ", "")
#     return text


# def preprocess_image(image_path):
#     """Preprocesses an image to improve OCR accuracy."""
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Error: Could not load image at {image_path}")
#         return None

#     # Convert to grayscale
#     gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Apply noise reduction
#     denoised_image = cv2.bilateralFilter(gray_image, 9, 75, 75)

#     # Apply Otsu's thresholding
#     _, otsu_thresh = cv2.threshold(
#         denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
#     )

#     return otsu_thresh


# # Process each image and extract Django code
# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     for image_file in os.listdir(IMAGE_DIR):
#         image_path = os.path.join(IMAGE_DIR, image_file)

#         print(f"🔍 Extracting text from: {image_file}...")

#         processed_image = preprocess_image(image_path)
#         if processed_image is None:
#             continue  # Skip this file if it couldn't be loaded

#         # Extract text using OCR
#         extracted_text = pytesseract.image_to_string(
#             processed_image, lang="eng", config=TESSERACT_CONFIG
#         )

#         # Clean the extracted text
#         cleaned_text = clean_extracted_text(extracted_text)

#         # Format the code using format_python_code
#         formatted_code = format_code(cleaned_text)

#         # Save formatted Django code
#         if formatted_code:  # Only save if there's valid code
#             f.write(f"# Code extracted from {image_file}\n")
#             f.write(formatted_code + "\n\n")

# print(f"✅ Django code extracted and saved to {OUTPUT_FILE}")


# import os
# import re

# import cv2
# import numpy as np
# import pytesseract
# from OCR_with_format.ocr_with_format import extract_text_with_format as format_code

# # Path to images and output file
# IMAGE_DIR = "output/images"
# OUTPUT_FILE = "output/extracted_code.py"

# # Configure Tesseract path (Only needed for Windows)
# TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH  # Remove this on
# Linux/macOS

# # Improved Tesseract OCR configuration
# TESSERACT_CONFIG = "--oem 3 --psm 4 -c preserve_interword_spaces=1"


# def preprocess_image(image_path):
#     """Preprocesses an image to improve OCR accuracy."""
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Error: Could not load image at {image_path}")
#         return None

#     # Convert to grayscale
#     gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Apply noise reduction
#     denoised_image = cv2.bilateralFilter(gray_image, 9, 75, 75)

#     # Apply Otsu's thresholding
#     _, otsu_thresh = cv2.threshold(
#         denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
#     )

#     return otsu_thresh


# # Process each image and extract Django code
# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     for image_file in os.listdir(IMAGE_DIR):
#         image_path = os.path.join(IMAGE_DIR, image_file)

#         print(f"🔍 Extracting text from: {image_file}...")

#         processed_image = preprocess_image(image_path)
#         if processed_image is None:
#             continue  # Skip this file if it couldn't be loaded

#         # Extract text using OCR
#         extracted_text = pytesseract.image_to_string(
#             processed_image, lang="eng", config=TESSERACT_CONFIG
#         )

#         # Format the code using format_code from OCR_with_format
#         try:
#             formatted_code = format_code(extracted_text)
#         except Exception as e:
#             print(f"Error formatting code: {e}")
#             formatted_code = None

#         # Save formatted Django code
#         if formatted_code:  # Only save if there's valid code
#             f.write(f"# Code extracted from {image_file}\n")
#             f.write(formatted_code + "\n\n")

# print(f"✅ Django code extracted and saved to {OUTPUT_FILE}")


# import os
# import subprocess

# # Path to images and output file
# IMAGE_DIR = "output/images"
# OUTPUT_FILE = "output/extracted_code.py"

# # Ensure the output directory exists
# os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# # Process each image in the directory
# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     for image_file in os.listdir(IMAGE_DIR):
#         image_path = os.path.join(IMAGE_DIR, image_file)

#         print(f"🔍 Extracting text from: {image_file}...")

#         # Run OCR_with_format using subprocess
#         try:
#             result = subprocess.run(
#                 ["OCR_with_format", image_path, "--method=with_format"],
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 text=True,
#             )

#             if result.returncode != 0:
#                 print(f"❌ Error processing {image_file}: {result.stderr.strip()}")
#                 continue

#             formatted_code = result.stdout.strip()

#             # Save the formatted code to the output file
#             if formatted_code:
#                 f.write(f"# Code extracted from {image_file}\n")
#                 f.write(formatted_code + "\n\n")

#         except subprocess.CalledProcessError as e:
#             print(f"❌ Error processing {image_file}: {e}")

# print(f"✅ Django code extracted and saved to {OUTPUT_FILE}")


import os
import subprocess

import cv2

# Path to images and output file
IMAGE_DIR = "output/images"
OUTPUT_FILE = "output/extracted_code.py"
PROCESSED_IMAGE_DIR = "output/processed_images"


# Ensure the output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)


# Process each image in the directory
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for image_file in os.listdir(IMAGE_DIR):
        image_path = os.path.join(IMAGE_DIR, image_file)

        # Load the image
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        from pre_procesing.preprocess import (
            preprocess_image,
            resizing2_300dpi_1500x_1500,
        )

        img = resizing2_300dpi_1500x_1500(img)

        # Save the image with DPI metadata
        processed_image_path = os.path.join(
            PROCESSED_IMAGE_DIR, image_file.replace(".jpg", "_processed.jpg")
        )
        cv2.imwrite(processed_image_path, img)

        print(f"🔍 Eastimating font size and applying CLAHE...")
        processed_image = preprocess_image(img)

        # Save processed image with correct extension
        print(f"✅ Saving processed image .....")
        processed_image_path = os.path.join(
            PROCESSED_IMAGE_DIR, image_file.replace(".png", "_processed.png")
        )
        cv2.imwrite(processed_image_path, processed_image)

        print(f"🔍 Running OCR on {image_file}...")
        # Run OCR_with_format using subprocess

        try:
            result = subprocess.run(
                ["OCR_with_format", processed_image_path, "--method=with_format"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode != 0:
                print(f"❌ OCR failed for {image_file}: {result.stderr.strip()}")
                continue

            formatted_code = result.stdout.strip()

            f.write(f"# Code extracted from {image_file}\n")
            f.write(formatted_code + "\n\n")

        except subprocess.CalledProcessError as e:
            print(f"❌ Error processing {image_file}: {e}")

print(f"✅ Django code extracted, improved, and saved to {OUTPUT_FILE}")
