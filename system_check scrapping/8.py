import os
import subprocess

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


def preprocess_image(image_path, output_path):
    """Enhanced preprocessing for code recognition"""
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    blurred = cv2.medianBlur(gray, 3)

    # Binarization
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Dilation and erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    cv2.imwrite(output_path, eroded)
    return eroded


def run_ocr(image):
    """Improved OCR processing with code-specific handling"""
    try:
        # Proper Tesseract configuration for code
        config = (
            "--psm 3 --oem 1 "  # Use psm 6 (single block)
            "preserve_interword_spaces=1"
        )

        results = pytesseract.image_to_data(
            image, output_type=Output.DICT, config=config, lang="eng"
        )
    except Exception as e:
        print(f"OCR Error: {e}")
        return None

    # Group words into lines using the top of the line
    line_dict = {}
    arrow_boxes = []  # To store dimensions of arrow symbols
    max_line_height = 0  # Initialize max_line_height

    for i in range(len(results["text"])):
        text = results["text"][i].strip()
        conf = int(results["conf"][i])

        # Use the top of the word as the line_key (for grouping)
        line_key = results["top"][i]
        if results["height"][i] > max_line_height:
            max_line_height = results["height"][i]

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

        line_dict[line_key]["texts"].append(text)
        line_dict[line_key]["lefts"].append(results["left"][i])
        line_dict[line_key]["heights"].append(results["height"][i])
        line_dict[line_key]["confs"].append(conf)
        line_dict[line_key]["widths"].append(results["width"][i])

        # Detect arrows and store their dimensions
        if any(symbol in text for symbol in ["->", "→"]):
            line_dict[line_key]["has_arrow"] = True
            arrow_boxes.append((results["width"][i], results["height"][i]))

            # Adjust confidence for low-confidence arrow detections
            if conf < 20:
                line_confs = line_dict[line_key]["confs"]
                median_conf = np.median(line_confs).item() if line_confs else 0
                if median_conf > 0:
                    results["conf"][i] = int(median_conf)
                    print(f"adjusting '{text}' conf from {conf} to {median_conf}")

        # Detect parentheses
        if any(symbol in text for symbol in ["(", ")"]):
            line_dict[line_key]["has_parentheses"] = True

    # Merge lines that are close together
    merged_line_dict = {}
    line_tolerance = max_line_height  # Maximum distance between lines to merge
    sorted_line_keys = sorted(line_dict.keys())

    for i, line_key in enumerate(sorted_line_keys):
        merged = False
        for merged_key in merged_line_dict.keys():
            if abs(line_key - merged_key) <= line_tolerance:
                # Merge the lines
                merged_line_dict[merged_key]["texts"].extend(
                    line_dict[line_key]["texts"]
                )
                merged_line_dict[merged_key]["lefts"].extend(
                    line_dict[line_key]["lefts"]
                )
                merged_line_dict[merged_key]["heights"].extend(
                    line_dict[line_key]["heights"]
                )
                merged_line_dict[merged_key]["confs"].extend(
                    line_dict[line_key]["confs"]
                )
                merged_line_dict[merged_key]["widths"].extend(
                    line_dict[line_key]["widths"]
                )
                merged_line_dict[merged_key]["has_arrow"] |= line_dict[line_key][
                    "has_arrow"
                ]
                merged_line_dict[merged_key]["has_parentheses"] |= line_dict[line_key][
                    "has_parentheses"
                ]
                merged = True
                break
        if not merged:
            # No nearby line to merge with, add as a new line
            merged_line_dict[line_key] = line_dict[line_key]

    # Reconstruct lines with proper ordering and filtering
    arrow_lines = []
    line_heights = []
    arrow_confidence = []
    raw_text = []

    for line in merged_line_dict.values():
        if line["has_arrow"] or line["has_parentheses"]:
            # Sort words by horizontal position
            sorted_words = list(zip(line["lefts"], line["texts"], line["confs"]))

            # Filter out non-code characters
            filtered_words = []
            filtered_conf = []
            for left, word, conf in sorted(sorted_words):
                if word.strip() and not word.isdigit():
                    filtered_words.append(word)
                    filtered_conf.append(conf)

            full_line = " ".join(filtered_words)

            # Combine fragmented arrows
            full_line = full_line.replace("- >", "->").replace("> -", "->")

            arrow_lines.append(full_line)
            line_heights.append(np.median(line["heights"]))

            # Find the confidence of '->' symbol and use it
            for word, conf in zip(filtered_words, filtered_conf):
                if any(symbol in word for symbol in ["->", "→"]):
                    arrow_confidence.append((word, conf))

            raw_text.append(full_line)  # new
    # Calculate medians
    median_box = (0, 0)
    if arrow_boxes:
        median_box = (
            int(np.median([w for w, h in arrow_boxes])),
            int(np.median([h for w, h in arrow_boxes])),
        )

    # Draw red boxes around lines with arrows or parentheses
    debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for line_key in merged_line_dict.keys():
        line = merged_line_dict[line_key]
        if line["has_arrow"] or line["has_parentheses"]:
            min_left = min(line["lefts"])
            min_top = line_key
            max_bottom = min_top + int(np.median(line["heights"]))
            # Compute max_left correctly
            max_left = max(
                left + width for left, width in zip(line["lefts"], line["widths"])
            )
            cv2.rectangle(
                debug_image, (min_left, min_top), (max_left, max_bottom), (0, 0, 255), 2
            )

    cv2.imwrite("debug_image_with_boxes.png", debug_image)

    return {
        "median_box": median_box,
        "median_line_height": np.median(line_heights).item() if line_heights else 0,
        "arrow_lines": arrow_lines,
        "arrow_confidence": arrow_confidence,
        # "all_lines": sorted(all_lines), # removed
        "raw_text": raw_text,  # new
        "raw_stats": {
            "total_arrows": len(arrow_boxes),
            "lines_with_arrows": len(arrow_lines),
        },
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

        # Make sure it's an image file (skip non-images)
        if not image_file.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp")
        ):
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
            print(f"total_arrows: {ocr_results['raw_stats']['total_arrows']}")
            print(f"lines with arrow: {ocr_results['raw_stats']['lines_with_arrows']}")
            print(f"Confidence for arrows: {ocr_results['arrow_confidence']}")

            # Save the extracted lines to the output file
            output_file.write(f"# Lines extracted from {image_file}\n")

            # filter
            filtered_code = filter_line(ocr_results["raw_text"])

            # Format the code with OCR_with_format
            for line in filtered_code:
                ocr_with_format_output = extract_code_from_image(processed_image_path)
                if ocr_with_format_output is not None:
                    output_file.write(f"{ocr_with_format_output}\n")

            output_file.write("\n")  # Add a separator between images
