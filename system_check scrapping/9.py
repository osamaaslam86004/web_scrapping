import os
import subprocess

import cv2
import numpy as np

import pytesseract
from pytesseract import Output

# Constants
IMAGE_DIR = "output/images"
OUTPUT_FILE = "output/extracted_code.py"
PROCESSED_IMAGE_DIR = "output/processed_images"

# Ensure directories exist
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)


def preprocess_image(image_path, output_path):
    """Enhances image for code recognition."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(output_path, processed)
    return processed


def run_ocr(image):
    """
    Performs OCR, detects arrows, and draws bounding boxes on lines with '->'.
    """
    try:
        config = (
            "--psm 11 --oem 3 "
            "-c tessedit_char_whitelist=->→{}[]()<>:;=+-*/\\\"'_.,|!@#%^& "
            "preserve_interword_spaces=1"
        )

        results = pytesseract.image_to_data(
            image, output_type=Output.DICT, config=config, lang="eng"
        )
    except Exception as e:
        print(f"OCR Error: {e}")
        return None

    # Group words by line
    line_dict = {}
    arrow_boxes = []
    extracted_text = []

    # Iterate to find each text
    for i in range(len(results["text"])):
        text = results["text"][i].strip()
        conf = int(results["conf"][i])

        # Calculate a line_key that is more robust to noise
        line_key = results["top"][i]

        if line_key not in line_dict:
            line_dict[line_key] = {
                "texts": [],
                "lefts": [],
                "heights": [],
                "confs": [],
                "widths": [],
                "has_arrow": False,
                "has_parentheses": False,
            }

        # Append data
        line_dict[line_key]["texts"].append(text)
        line_dict[line_key]["lefts"].append(results["left"][i])
        line_dict[line_key]["heights"].append(results["height"][i])
        line_dict[line_key]["confs"].append(conf)
        line_dict[line_key]["widths"].append(results["width"][i])

        # Detect arrows
        if any(symbol in text for symbol in ["->", "→"]):
            line_dict[line_key]["has_arrow"] = True
            arrow_boxes.append((results["width"][i], results["height"][i]))

        # Detect parentheses
        if any(symbol in text for symbol in ["(", ")"]):
            line_dict[line_key]["has_parentheses"] = True

    # Now reconstruct the lines and determine what to draw on
    arrow_lines = []
    line_heights = []
    debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    line_count = 0

    for line_key, line in line_dict.items():
        if line["has_arrow"] or line["has_parentheses"]:
            # We're drawing a bounding box on it
            min_left = min(line["lefts"])
            max_left = max(
                [left + width for left, width in zip(line["lefts"], line["widths"])]
            )
            min_top = line_key
            max_bottom = min_top + max(line["heights"])

            # Add tolerance
            box_tolerance = 5
            min_top -= box_tolerance  # add tolerance to the top and bottom so the box encapsulates the full text
            max_bottom += 2 * box_tolerance

            cv2.rectangle(
                debug_image,
                (min_left, min_top),
                (max_left, max_bottom),
                (0, 0, 255),
                2,
            )
            line_count += 1

            # Sort the words by their left positions
            sorted_words = [w for _, w in sorted(zip(line["lefts"], line["texts"]))]
            full_line = " ".join(sorted_words)
            arrow_lines.append(full_line)
            line_heights.append(np.median(line["heights"]))
            extracted_text.append(line)

    # Write image to file
    cv2.imwrite("debug_image_with_boxes.png", debug_image)

    # Calculate median box
    median_box = (
        np.median([w for w, h in arrow_boxes]).item() if arrow_boxes else (0, 0),
        np.median([h for w, h in arrow_boxes]).item() if arrow_boxes else (0, 0),
    )

    return {
        "median_box": median_box,
        "median_line_height": np.median(line_heights).item() if line_heights else 0,
        "arrow_lines": arrow_lines,
        "raw_stats": {"total_arrows": len(arrow_boxes), "line_count": line_count},
        "extracted_text": extracted_text,
    }


def filter_line(lines):
    """remove number at the begining of line"""
    filtered_lines = []
    for line in lines:
        words = line.split()
        if words and words[0].isdigit() and len(words[0]) <= 3:
            filtered_lines.append(" ".join(words[1:]))
        else:
            filtered_lines.append(line)
    return filtered_lines


def extract_code_from_image(image_path):
    """Extracts code from an image using OCR_with_format."""
    print(f"🔍 Running OCR_with_format on: {image_path}")

    # Improved Tesseract OCR configuration
    TESSERACT_CONFIG = "--oem 3 --psm 6 -c preserve_interword_spaces=1"

    try:
        result = subprocess.run(
            [
                "OCR_with_format",
                image_path,
                "--method=with_format",
                "--thresholding_method=all",
                f"--tesseract_args={TESSERACT_CONFIG}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            print(f"❌ OCR failed: {result.stderr.strip()}")
            return None

        extracted_code = result.stdout.strip()

        # Debug: Check OCR output before saving
        # print(f"📜 Extracted code preview:\n{extracted_code}\n")
        return extracted_code

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during OCR processing: {e}")
        return None


with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
    # Process all images in the directory
    for image_file in os.listdir(IMAGE_DIR):
        image_path = os.path.join(IMAGE_DIR, image_file)

        # Ensure it's an image file
        if not image_file.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp")
        ):
            print(f"Skipping non-image file: {image_file}")
            continue

        processed_image_path = os.path.join(
            PROCESSED_IMAGE_DIR, f"{image_file}_processed.jpg"
        )

        # Preprocess the image
        processed_image = preprocess_image(image_path, processed_image_path)

        # Process with OCR if the image was successfully preprocessed
        if processed_image is not None:
            ocr_results = run_ocr(processed_image)

            if ocr_results:
                print(f"Median arrow box: {ocr_results['median_box']}")
                print(f"Median line height: {ocr_results['median_line_height']:.1f}px")
                print("Lines containing '->':")
                for line in ocr_results["arrow_lines"]:
                    print(f"- {line}")
                print(f"total_arrows: {ocr_results['raw_stats']['total_arrows']}")
                print(f"line_count: {ocr_results['raw_stats']['line_count']}")
                print(ocr_results)

                # Save the extracted lines to the output file
                output_file.write(f"# Lines extracted from {image_file}\n")

                # filter
                filtered_code = filter_line(ocr_results["extracted_text"])

                # Format the code with OCR_with_format
                for line in filtered_code:
                    ocr_with_format_output = extract_code_from_image(
                        processed_image_path
                    )
                    if ocr_with_format_output is not None:
                        output_file.write(f"{ocr_with_format_output}\n")

                output_file.write("\n")  # Add a separator between images
