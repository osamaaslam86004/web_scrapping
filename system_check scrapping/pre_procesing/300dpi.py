# import os

# import cv2
# import easyocr
# import numpy as np
# from PIL import Image

# # Path to images and output file
# IMAGE_DIR = "output/images"
# OUTPUT_FILE = "output/extracted_code.py"

# # Create an EasyOCR reader with GPU disabled (force CPU mode)
# reader = easyocr.Reader(["en"], gpu=False)  # Ensure CPU compatibility

# # Ensure output directory exists
# os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


# def convert_to_300dpi(image_path):
#     """
#     Converts an image to 300 DPI without modifying its content.
#     """
#     img = Image.open(image_path)
#     img.save(image_path, dpi=(300, 300))
#     return image_path


# # Process each image and extract Django code
# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     for image_file in os.listdir(IMAGE_DIR):
#         image_path = os.path.join(IMAGE_DIR, image_file)
#         print(f"🔍 Processing Image: {image_file}...")

#         # Convert image to 300 DPI
#         image_path = convert_to_300dpi(image_path)

#         # Load image (original, without modifications)
#         img = cv2.imread(image_path)
#         if img is None:
#             print(
#                 f"❌ Error: Could not load {image_file}. Check if it's a valid image file."
#             )
#             continue  # Skip this file

#         # Extract text using EasyOCR with optimized parameters
#         result = reader.readtext(
#             img,
#             detail=0,
#             paragraph=True,
#             decoder="beamsearch",
#             contrast_ths=0.1,
#             adjust_contrast=True,
#         )
#         extracted_text = "\n".join(result)

#         # Debug: Print extracted text
#         print(f"📜 Extracted text from {image_file}:\n{extracted_text}\n")

#         # Save extracted text to file
#         f.write(f"# Code extracted from {image_file}\n")
#         f.write(extracted_text + "\n\n")

# print(f"✅ Django code extracted and saved to {OUTPUT_FILE}")


# import os

# import cv2
# import easyocr
# import numpy as np
# from PIL import Image

# # Path to images and output file
# IMAGE_DIR = "output/images"
# OUTPUT_FILE = "output/extracted_code.py"

# # Create an EasyOCR reader with GPU disabled (force CPU mode)
# reader = easyocr.Reader(["en"], gpu=False)  # Ensure CPU compatibility

# # Ensure output directory exists
# os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


# def convert_to_300dpi(image_path):
#     """
#     Converts an image to 300 DPI without modifying its content.
#     """
#     img = Image.open(image_path)
#     img.save(image_path, dpi=(300, 300))
#     return image_path


# # Process each image and extract Django code
# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     for image_file in os.listdir(IMAGE_DIR):
#         image_path = os.path.join(IMAGE_DIR, image_file)
#         print(f"🔍 Processing Image: {image_file}...")

#         # Convert image to 300 DPI
#         image_path = convert_to_300dpi(image_path)

#         # Load image (original, without modifications)
#         img = cv2.imread(image_path)
#         if img is None:
#             print(
#                 f"❌ Error: Could not load {image_file}. Check if it's a valid image file."
#             )
#             continue  # Skip this file

#         # Extract text using EasyOCR with optimized parameters
#         result = reader.readtext(
#             img,
#             detail=1,
#             paragraph=False,
#             decoder="wordbeamsearch",
#             contrast_ths=0.5,
#             adjust_contrast=0.5,
#         )
#         extracted_text = "\n".join([text[1] for text in result])

#         # Debug: Print extracted text
#         print(f"📜 Extracted text from {image_file}:\n{extracted_text}\n")

#         # Save extracted text to file
#         f.write(f"# Code extracted from {image_file}\n")
#         f.write(extracted_text + "\n\n")

# print(f"✅ Django code extracted and saved to {OUTPUT_FILE}")


# import os

# import cv2
# import easyocr
# import numpy as np
# from PIL import Image, ImageEnhance

# # Path to images and output file
# IMAGE_DIR = "output/images"
# OUTPUT_FILE = "output/extracted_code.py"

# # Create an EasyOCR reader with GPU disabled (force CPU mode)
# reader = easyocr.Reader(["en"], gpu=False)  # Ensure CPU compatibility

# # Ensure output directory exists
# os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


# def preprocess_image(image_path):
#     """
#     Enhances image contrast and ensures it is 300 DPI for better OCR accuracy.
#     """
#     img = Image.open(image_path)
#     enhancer = ImageEnhance.Contrast(img)
#     img = enhancer.enhance(2.0)  # Increase contrast
#     img.save(image_path, dpi=(300, 300))
#     return image_path


# # Process each image and extract Django code
# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     for image_file in os.listdir(IMAGE_DIR):
#         image_path = os.path.join(IMAGE_DIR, image_file)
#         print(f"🔍 Processing Image: {image_file}...")

#         # Preprocess image (enhance contrast & set DPI)
#         image_path = preprocess_image(image_path)

#         # Load image (grayscale for better OCR)
#         img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             print(
#                 f"❌ Error: Could not load {image_file}. Check if it's a valid image file."
#             )
#             continue  # Skip this file

#         # Extract text using EasyOCR with optimized parameters
#         result = reader.readtext(
#             img,
#             detail=1,
#             paragraph=True,
#             contrast_ths=0.7,
#             adjust_contrast=0.7,
#             text_threshold=0.6,
#         )
#         extracted_text = "\n".join([text[1] for text in result])

#         # Debug: Print extracted text
#         print(f"📜 Extracted text from {image_file}:\n{extracted_text}\n")

#         # Save extracted text to file
#         f.write(f"# Code extracted from {image_file}\n")
#         f.write(extracted_text + "\n\n")

# print(f"✅ Django code extracted and saved to {OUTPUT_FILE}")


import os

import cv2
import easyocr
import numpy as np
from PIL import Image, ImageEnhance

# Path to images and output file
IMAGE_DIR = "output/images"
OUTPUT_FILE = "output/extracted_code.py"

# Create an EasyOCR reader with GPU disabled (force CPU mode)
reader = easyocr.Reader(["en"], gpu=False)  # Ensure CPU compatibility

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


def preprocess_image(image_path):
    """
    Enhances image contrast and ensures it is 300 DPI for better OCR accuracy.
    """
    img = Image.open(image_path).convert("RGB")  # Ensure RGB mode
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)  # Increase contrast
    img.save(image_path, dpi=(300, 300))
    return image_path


# Process each image and extract Django code
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for image_file in os.listdir(IMAGE_DIR):
        image_path = os.path.join(IMAGE_DIR, image_file)
        print(f"🔍 Processing Image: {image_file}...")

        # Preprocess image (enhance contrast & set DPI)
        image_path = preprocess_image(image_path)

        # Load image (grayscale for better OCR)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(
                f"❌ Error: Could not load {image_file}. Check if it's a valid image file."
            )
            continue  # Skip this file

        # Extract text using EasyOCR with optimized parameters
        result = reader.readtext(
            img,
            detail=1,
            paragraph=True,
            contrast_ths=0.7,
            adjust_contrast=0.7,
            text_threshold=0.6,
        )
        extracted_text = "\n".join([text[1] for text in result])

        # Debug: Print extracted text
        print(f"📜 Extracted text from {image_file}:\n{extracted_text}\n")

        # Save extracted text to file
        f.write(f"# Code extracted from {image_file}\n")
        f.write(extracted_text + "\n\n")

print(f"✅ Django code extracted and saved to {OUTPUT_FILE}")


# import cv2
# import numpy as np
# from PIL import Image

# # Load the uploaded image
# image_path = "/mnt/data/check-19ece316.png"
# img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale

# # Analyze basic properties
# image_properties = {
#     "shape": img.shape,
#     "mean_intensity": np.mean(img),
#     "std_intensity": np.std(img),
#     "min_intensity": np.min(img),
#     "max_intensity": np.max(img)
# }

# image_properties


# {'shape': (1350, 1676),
#  'mean_intensity': 245.98191240166182,
#  'std_intensity': 42.28100038261876,
#  'min_intensity': 0,
#  'max_intensity': 255}
