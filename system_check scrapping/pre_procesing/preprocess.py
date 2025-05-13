import cv2

import pytesseract


def estimate_font_size(img):
    """
    Estimates the font size of text in an image by analyzing bounding box heights.
    Uses OpenCV + Tesseract for lightweight processing.
    """

    # Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img

    # Run Tesseract OCR with bounding box detection
    boxes = pytesseract.image_to_boxes(gray)

    # Extract text heights from bounding boxes
    font_sizes = []
    for box in boxes.splitlines():
        b = box.split()
        x, y, w, h = map(
            int, [b[1], b[2], b[3], b[4]]
        )  # Extract bounding box coordinates
        text_height = h - y  # Calculate height of text box
        font_sizes.append(text_height)

    # Calculate average detected font size
    if font_sizes:
        avg_font_size = sum(font_sizes) / len(font_sizes)
        print(f"📏 Estimated Font Size: {avg_font_size:.2f} px")

        # Check if font size is below 16px
        if avg_font_size < 16:
            print(
                "⚠️ Warning: Font size is too small for accurate OCR! Consider increasing it."
            )
        else:
            print("✅ Font size is good for OCR.")
        return avg_font_size, gray
    else:
        print("⚠️ No text detected!")
        return None


def resizing2_300dpi_1500x_1500(img):
    # Original image dimensions
    original_height, original_width = img.shape[:2]

    # Define target DPI and scale factor
    target_dpi = 300
    original_dpi = 96  # Assuming input image has 96 DPI
    dpi_scale_factor = target_dpi / original_dpi  # 300 / 96 = 3.125

    # Calculate new dimensions while maintaining aspect ratio
    new_width = int(original_width / dpi_scale_factor)
    new_height = int(original_height / dpi_scale_factor)

    # Ensure it fits within 1500x1500 px while keeping aspect ratio
    max_size = 1500
    if new_width > max_size or new_height > max_size:
        scale_factor = max_size / max(new_width, new_height)
        new_width = int(new_width * scale_factor)
        new_height = int(new_height * scale_factor)

    # Resize the image
    resized_img = cv2.resize(
        img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
    )
    print(f"✅ Image resized to {new_width}x{new_height} pixels at {target_dpi} DPI")

    return resized_img


def apply_clahe(gray):
    """
    Enhances image quality for better OCR by:
    - Applying CLAHE (Contrast Enhancement)
    """

    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    processed_img = clahe.apply(gray)

    return processed_img


def preprocess_image(img):
    font_size, gray = estimate_font_size(img)
    # processed_img = apply_clahe(gray)
    return gray
