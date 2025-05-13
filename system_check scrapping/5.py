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


def adjust_gamma(image, gamma=1.8):
    """
    Adjusts brightness of the entire image using Gamma Correction.
    """
    inv_gamma = 1.0 / gamma
    table = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def preprocess_image(image_path, output_path):
    """
    Enhances brightness and contrast while keeping text sharp for OCR.
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

    # 2. Apply Gamma Correction to brighten the entire image
    # brightened = adjust_gamma(img, gamma=0.8)  # Increase brightness

    brightened = img

    # 3. Convert to Grayscale (after brightness enhancement)
    gray = cv2.cvtColor(brightened, cv2.COLOR_BGR2GRAY)

    # 4. Improve Contrast using CLAHE (Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)

    # 5. Save Processed Image
    cv2.imwrite(output_path, contrast_enhanced)
    print(f"✅ Processed image saved: {output_path}")

    return output_path


def run_ocr(image_path, output_code_path):
    """
    Runs OCR-with-format on the preprocessed image.
    """
    print(f"🔍 Running OCR on: {image_path}")

    # Improved Tesseract OCR configuration
    TESSERACT_CONFIG = "oem 1 --psm 4 -c preserve_interword_spaces=1"

    try:
        result = subprocess.run(
            [
                "OCR_with_format",
                image_path,
                "--method=with_format",
                "--thresholding_method=all",
                "--tesseract_args=f'{TESSERACT_CONFIG}'",
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
    if not image_file.lower().endswith(
        (".png", ".jpg", ".jpeg", ".tiff", ".bmp", "webp")
    ):
        print(f"⚠️ Skipping non-image file: {image_file}")
        continue

    processed_image_path = os.path.join(
        PROCESSED_IMAGE_DIR, f"{image_file}_processed.jpg"
    )

    # Step 1: Preprocess Image
    # processed_image_path = preprocess_image(image_path, processed_image_path)

    if processed_image_path is not None:
        # Step 2: Run OCR on Processed Image
        run_ocr(image_path, OUTPUT_FILE)
