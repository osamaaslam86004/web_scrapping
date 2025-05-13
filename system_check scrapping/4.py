# import os
# import subprocess

# import cv2
# import numpy as np
# from PIL import Image

# # Paths
# IMAGE_DIR = "output/images"
# OUTPUT_FILE = "output/extracted_code.py"
# PROCESSED_IMAGE_DIR = "output/processed_images"

# # Ensure output directories exist
# os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
# os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)


# def preprocess_image(image_path, output_path):
#     """
#     Prepares an image for OCR by normalizing it:
#     - Converts to grayscale
#     - Applies skew correction
#     - Resizes to consistent dimensions (300 DPI)
#     - Enhances contrast and applies adaptive thresholding
#     """
#     print(f"\n🔄 Processing: {image_path}")

#     # 1. Load Image with Fallback
#     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     if img is None:
#         try:
#             pil_img = Image.open(image_path).convert("RGB")
#             img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
#         except Exception as e:
#             print(f"❌ Failed to load image: {e}")
#             return None

#     print(f"📏 Original Image shape: {img.shape}")

#     rotated = img
#     # 4. Resize to Consistent DPI (300 DPI)
#     scale_factor = 300 / 96  # Scale image to 300 DPI
#     new_width = int(rotated.shape[1] * scale_factor)
#     new_height = int(rotated.shape[0] * scale_factor)
#     resized_img = cv2.resize(
#         rotated, (new_width, new_height), interpolation=cv2.INTER_CUBIC
#     )

#     # 6. Save Processed Image
#     cv2.imwrite(output_path, resized_img)
#     print(f"✅ Processed image saved: {output_path}")

#     return output_path


# def run_ocr(image_path, output_code_path):
#     """
#     Runs OCR-with-format on the preprocessed image.
#     """
#     print(f"🔍 Running OCR on: {image_path}")

#     # Improved Tesseract OCR configuration
#     TESSERACT_CONFIG = "--psm 15 -c preserve_interword_spaces=1"

#     try:
#         result = subprocess.run(
#             [
#                 "OCR_with_format",
#                 image_path,
#                 "--method=with_format",
#                 "--thresholding_method=all",
#                 "--tesseract_args=f'{TESSERACT_CONFIG}'",
#             ],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True,
#         )

#         if result.returncode != 0:
#             print(f"❌ OCR failed: {result.stderr.strip()}")
#             return

#         extracted_code = result.stdout.strip()

#         # Save extracted code
#         with open(output_code_path, "a", encoding="utf-8") as f:
#             f.write(f"\n# Code extracted from {image_path}\n")
#             f.write(extracted_code + "\n")

#         print(f"✅ Extracted code saved to: {output_code_path}")

#     except subprocess.CalledProcessError as e:
#         print(f"❌ Error during OCR processing: {e}")


# # Process all images in the directory
# for image_file in os.listdir(IMAGE_DIR):
#     image_path = os.path.join(IMAGE_DIR, image_file)

#     # Make sure it's an image file (skip non-images)
#     if not image_file.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
#         print(f"⚠️ Skipping non-image file: {image_file}")
#         continue

#     processed_image_path = os.path.join(
#         PROCESSED_IMAGE_DIR, f"{image_file}_processed.jpg"
#     )

#     # Step 1: Preprocess Image with Normalization
#     processed_image_path = preprocess_image(image_path, processed_image_path)

#     if processed_image_path is not None:  # Run OCR only if processing succeeded
#         # Step 2: Run OCR on Processed Image
#         run_ocr(processed_image_path, OUTPUT_FILE)


import os
import subprocess

import cv2
import numpy as np
from PIL import Image

# Paths
IMAGE_DIR = "output/images"
OUTPUT_FILE = "output/extracted_code.py"
PROCESSED_IMAGE_DIR = "output/processed_images"

# Ensure output directories exist
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)


def preprocess_image(image_path, output_path):
    """
    Prepares an image for OCR by making the text bolder using dilation.
    """
    print(f"\n🔄 Processing: {image_path}")

    # 1. Load Image with Fallback
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        try:
            pil_img = Image.open(image_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"❌ Failed to load image: {e}")
            return None

    print(f"📏 Original Image shape: {img.shape}")

    # 4. Dilation (Making Text Bolder)
    kernel = np.ones((3, 3), np.uint8)  # Adjust kernel size as needed
    dilated_img = cv2.dilate(img, kernel, iterations=1)

    # 5. Save Processed Image
    cv2.imwrite(output_path, dilated_img)
    print(f"✅ Processed image saved: {output_path}")

    return output_path  # Return the path, NOT the NumPy array


def run_ocr(image_path, output_code_path):
    """
    Runs OCR-with-format on the preprocessed image.
    """
    print(f"🔍 Running OCR on: {image_path}")

    # Improved Tesseract OCR configuration
    TESSERACT_CONFIG = "oem 3 --psm 6 -c preserve_interword_spaces=1"

    try:
        result = subprocess.run(
            [
                "OCR_with_format",
                image_path,
                "--method=with_format",
                "--thresholding_method=all",
                # "--tesseract_args=f'{TESSERACT_CONFIG}'",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            print(f"❌ OCR failed: {result.stderr.strip()}")
            return

        extracted_code = result.stdout.strip()

        # Debug: Check OCR output before saving
        print(f"📜 Extracted code preview:\n{extracted_code}\n")

        # Save extracted code
        with open(output_code_path, "a", encoding="utf-8") as f:
            f.write(f"\n# Code extracted from {image_path}\n")
            f.write(extracted_code + "\n")

        print(f"✅ Extracted code saved to: {output_code_path}")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during OCR processing: {e}")


# Process all images in the directory
for image_file in os.listdir(IMAGE_DIR):
    image_path = os.path.join(IMAGE_DIR, image_file)

    # Make sure it's an image file (skip non-images)
    if not image_file.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
        print(f"⚠️ Skipping non-image file: {image_file}")
        continue

    processed_image_path = os.path.join(
        PROCESSED_IMAGE_DIR, f"{image_file}_processed.jpg"
    )

    # Step 1: Preprocess Image
    processed_image_path = preprocess_image(image_path, processed_image_path)

    if processed_image_path is not None:  # ✅ Fixed condition check
        # Step 2: Run OCR on Processed Image
        run_ocr(processed_image_path, OUTPUT_FILE)
