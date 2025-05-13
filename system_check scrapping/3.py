import os

import cv2
import numpy as np

import pytesseract
from pytesseract import Output

# Paths
IMAGE_DIR = "output/images"
OUTPUT_FILE = "output/extracted_code.py"
PROCESSED_IMAGE_DIR = "output/processed_images"

# Ensure output directories exist
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)


def preprocess_image(img, output_path):
    """ "
    Preprocess images with OpenCV to enhance clarity:
    """
    # Read the image
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imwrite(output_path, thresh)  # Save the processed image
    return thresh


def run_ocr(image):
    """
    Enhanced version with better line grouping and symbol detection
    """

    try:
        results = pytesseract.image_to_data(
            image,
            output_type=Output.DICT,
            config="--psm 6 preserve_interword_spaces=1",
        )
    except pytesseract.pytesseract.TesseractError as e:
        print(f"Tesseract error: {e}")
        return None

    # Group words into lines with tolerance for vertical positioning
    line_tolerance = 5  # pixels
    lines = {}
    arrow_boxes = []
    line_metrics = []

    # First pass: group words into lines and detect arrows
    for i in range(len(results["text"])):
        text = results["text"][i].strip()
        if not text:
            continue

        # Find existing line or create new one
        current_top = results["top"][i]
        line_key = next(
            (key for key in lines.keys() if abs(key - current_top) <= line_tolerance),
            None,
        )

        if line_key is None:
            line_key = current_top
            lines[line_key] = {
                "text": [],
                "left": [],
                "height": results["height"][i],
                "contains_arrow": False,
            }

        # Store word information
        lines[line_key]["text"].append(text)
        lines[line_key]["left"].append(results["left"][i])

        # Detect arrows with flexible matching
        if any(symbol in text for symbol in ["->", "→", "-->"]):
            arrow_boxes.append((results["width"][i], results["height"][i]))
            lines[line_key]["contains_arrow"] = True

    # Second pass: reconstruct lines with proper ordering
    arrow_lines = []
    for line in lines.values():
        if line["contains_arrow"]:
            # Sort words by left position
            sorted_words = [word for _, word in sorted(zip(line["left"], line["text"]))]
            full_line = " ".join(sorted_words)
            arrow_lines.append(full_line)
            line_metrics.append(line["height"])

    # Calculate medians
    median_box = (
        np.median([w for w, h in arrow_boxes]).item() if arrow_boxes else 0,
        np.median([h for w, h in arrow_boxes]).item() if arrow_boxes else 0,
    )

    return {
        "median_box": median_box,
        "median_line_height": np.median(line_metrics).item() if line_metrics else 0,
        "arrow_lines": arrow_lines,
        "raw_stats": {
            "total_arrows_found": len(arrow_boxes),
            "lines_with_arrows": len(arrow_lines),
        },
    }


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
    processed_image = preprocess_image(image_path, processed_image_path)

    ocr_results = run_ocr(processed_image)

    if ocr_results:  # Run OCR only if processing succeeded
        # Step 2: Run OCR on Processed Image
        print(f"Median arrow box: {ocr_results['median_box']}")
        print(f"Median line height: {ocr_results['median_line_height']:.1f}px")
        print("Lines containing '->':")
        for line in ocr_results["arrow_lines"]:
            print(f"- {line}")
