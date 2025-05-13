# # def preprocess_image(image_path, output_path):
# # """
# # Prepares an image for OCR by:
# # - Resizing while maintaining aspect ratio
# # """
# # print(f"🔍 Processing: {image_path}")

# # # Load the image (Make sure it's a valid file)
# # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # # Get original dimensions
# # original_width, original_height = img.shape[1], img.shape[0]

# # # Resize dynamically to 300 DPI
# # target_dpi = 600
# # scale_factor = target_dpi / 96  # Original DPI is 96
# # new_width = int(original_width * scale_factor)
# # new_height = int(original_height * scale_factor)

# # if new_width > 2500 or new_height > 2500:  # Keep aspect ratio
# #     scale_factor = 2500 / max(new_width, new_height)
# #     new_width = int(new_width * scale_factor)
# #     new_height = int(new_height * scale_factor)

# # resized_img = cv2.resize(
# #     img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
# # )

# # from pre_procesing.preprocess import estimate_font_size

# # font_size, gray = estimate_font_size(resized_img)

# # # Save processed image with 300 DPI
# # pil_img = Image.fromarray(resized_img)
# # pil_img.save(output_path, dpi=(600, 600))

# # print(f"✅ Image saved: {output_path} (Optimized for OCR)")
# # return output_path


# # import os
# # import subprocess

# # import cv2
# # import numpy as np
# # from PIL import Image

# # # Paths
# # IMAGE_DIR = "output/images"
# # OUTPUT_FILE = "output/extracted_code.py"
# # PROCESSED_IMAGE_DIR = "output/processed_images"

# # # Ensure output directories exist
# # os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
# # os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)


# # def preprocess_image(image_path, output_path):
# #     """Loads and preprocesses an image for OCR."""
# #     print(f"\n🔄 Processing: {image_path}")

# #     # 1. Load Image with Fallback
# #     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# #     if img is None:
# #         try:
# #             pil_img = Image.open(image_path).convert("RGB")
# #             img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
# #         except Exception as e:
# #             print(f"❌ Failed to load image: {e}")
# #             return None

# #     print(f"📏 Image shape: {img.shape}")

# #     # 2. Convert to Grayscale
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# #     # 3. Apply CLAHE (Contrast Enhancement)
# #     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
# #     enhanced_img = clahe.apply(gray)

# #     # 4. Apply Adaptive Thresholding
# #     _, thresh = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# #     # 5. Save Processed Image
# #     cv2.imwrite(output_path, thresh)
# #     print(f"✅ Processed image saved: {output_path}")
# #     return thresh


# # def run_ocr(image_path, output_code_path):
# #     """
# #     Runs OCR-with-format on the preprocessed image.
# #     """
# #     print(f"🔍 Running OCR on: {image_path}")

# #     try:
# #         result = subprocess.run(
# #             ["OCR_with_format", image_path, "--method=with_format"],
# #             stdout=subprocess.PIPE,
# #             stderr=subprocess.PIPE,
# #             text=True,
# #         )

# #         if result.returncode != 0:
# #             print(f"❌ OCR failed: {result.stderr.strip()}")
# #             return

# #         extracted_code = result.stdout.strip()

# #         # Save extracted code
# #         with open(output_code_path, "w", encoding="utf-8") as f:
# #             f.write(extracted_code + "\n")

# #         print(f"✅ Extracted code saved to: {output_code_path}")

# #     except subprocess.CalledProcessError as e:
# #         print(f"❌ Error during OCR processing: {e}")


# # # Process all images in the directory
# # for image_file in os.listdir(IMAGE_DIR):
# #     image_path = os.path.join(IMAGE_DIR, image_file)

# #     # Make sure it's an image file (skip non-images)
# #     if not image_file.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
# #         print(f"⚠️ Skipping non-image file: {image_file}")
# #         continue

# #     processed_image_path = os.path.join(
# #         PROCESSED_IMAGE_DIR, f"{image_file}_processed.jpg"
# #     )

# #     # Step 1: Preprocess Image
# #     processed_image = preprocess_image(image_path, processed_image_path)

# #     if processed_image is not None:  # Run OCR only if processing succeeded
# #         # Step 2: Run OCR on Processed Image
# #         run_ocr(processed_image, OUTPUT_FILE)


# # import os
# # import subprocess

# # import cv2
# # import numpy as np
# # from PIL import Image

# # # Paths
# # IMAGE_DIR = "output/images"
# # OUTPUT_FILE = "output/extracted_code.py"
# # PROCESSED_IMAGE_DIR = "output/processed_images"

# # # Ensure output directories exist
# # os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
# # os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)


# # def preprocess_image(image_path, output_path):
# #     """Loads and preprocesses an image for OCR."""
# #     print(f"\n🔄 Processing: {image_path}")

# #     # 1. Load Image with Fallback
# #     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# #     if img is None:
# #         try:
# #             pil_img = Image.open(image_path).convert("RGB")
# #             img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
# #         except Exception as e:
# #             print(f"❌ Failed to load image: {e}")
# #             return None

# #     print(f"📏 Image shape: {img.shape}")

# #     # 2. Convert to Grayscale
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# #     # 3. Apply CLAHE (Contrast Enhancement)
# #     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
# #     enhanced_img = clahe.apply(gray)

# #     # 4. Apply Adaptive Thresholding
# #     _, thresh = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# #     # 5. Save Processed Image
# #     cv2.imwrite(output_path, thresh)
# #     print(f"✅ Processed image saved: {output_path}")

# #     return output_path  # Return the path, NOT the NumPy array


# # def run_ocr(image_path, output_code_path):
# #     """
# #     Runs OCR-with-format on the preprocessed image.
# #     """
# #     print(f"🔍 Running OCR on: {image_path}")

# #     try:
# #         result = subprocess.run(
# #             ["OCR_with_format", image_path, "--method=with_format"],
# #             stdout=subprocess.PIPE,
# #             stderr=subprocess.PIPE,
# #             text=True,
# #         )

# #         if result.returncode != 0:
# #             print(f"❌ OCR failed: {result.stderr.strip()}")
# #             return

# #         extracted_code = result.stdout.strip()

# #         # Save extracted code
# #         with open(output_code_path, "a", encoding="utf-8") as f:
# #             f.write(f"\n# Code extracted from {image_path}\n")
# #             f.write(extracted_code + "\n")

# #         print(f"✅ Extracted code saved to: {output_code_path}")

# #     except subprocess.CalledProcessError as e:
# #         print(f"❌ Error during OCR processing: {e}")


# # # Process all images in the directory
# # for image_file in os.listdir(IMAGE_DIR):
# #     image_path = os.path.join(IMAGE_DIR, image_file)

# #     # Make sure it's an image file (skip non-images)
# #     if not image_file.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
# #         print(f"⚠️ Skipping non-image file: {image_file}")
# #         continue

# #     processed_image_path = os.path.join(
# #         PROCESSED_IMAGE_DIR, f"{image_file}_processed.jpg"
# #     )

# #     # Step 1: Preprocess Image
# #     processed_image_path = preprocess_image(image_path, processed_image_path)

# #     if processed_image_path is not None:  # ✅ Fixed condition check
# #         # Step 2: Run OCR on Processed Image
# #         run_ocr(processed_image_path, OUTPUT_FILE)


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
#     """Loads and preprocesses an image for OCR."""
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

#     print(f"📏 Image shape: {img.shape}")

#     # 2. Convert to Grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # 3. Increase DPI for Better OCR
#     scale_factor = 300 / 96  # Scale image to 300 DPI
#     new_width = int(gray.shape[1] * scale_factor)
#     new_height = int(gray.shape[0] * scale_factor)
#     high_res = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

#     # 8. Save Processed Image
#     cv2.imwrite(output_path, high_res)
#     print(f"✅ Processed image saved: {output_path}")

#     return output_path  # Return the path, NOT the NumPy array


# def run_ocr(image_path, output_code_path):
#     """
#     Runs OCR-with-format on the preprocessed image.
#     """
#     print(f"🔍 Running OCR on: {image_path}")

#     try:
#         result = subprocess.run(
#             [
#                 "OCR_with_format",
#                 image_path,
#                 "--method=with_format --thresholding_method=all",
#             ],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True,
#         )

#         if result.returncode != 0:
#             print(f"❌ OCR failed: {result.stderr.strip()}")
#             return

#         extracted_code = result.stdout.strip()

#         # Debug: Check OCR output before saving
#         print(f"📜 Extracted code preview:\n{extracted_code}\n")

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

#     # Step 1: Preprocess Image
#     processed_image_path = preprocess_image(image_path, processed_image_path)

#     if processed_image_path is not None:  # ✅ Fixed condition check
#         # Step 2: Run OCR on Processed Image
#         run_ocr(processed_image_path, OUTPUT_FILE)


# def preprocess_image(image_path, output_path):
# """
# Prepares an image for OCR by:
# - Resizing while maintaining aspect ratio
# """
# print(f"🔍 Processing: {image_path}")

# # Load the image (Make sure it's a valid file)
# img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # Get original dimensions
# original_width, original_height = img.shape[1], img.shape[0]

# # Resize dynamically to 300 DPI
# target_dpi = 600
# scale_factor = target_dpi / 96  # Original DPI is 96
# new_width = int(original_width * scale_factor)
# new_height = int(original_height * scale_factor)

# if new_width > 2500 or new_height > 2500:  # Keep aspect ratio
#     scale_factor = 2500 / max(new_width, new_height)
#     new_width = int(new_width * scale_factor)
#     new_height = int(new_height * scale_factor)

# resized_img = cv2.resize(
#     img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
# )

# from pre_procesing.preprocess import estimate_font_size

# font_size, gray = estimate_font_size(resized_img)

# # Save processed image with 300 DPI
# pil_img = Image.fromarray(resized_img)
# pil_img.save(output_path, dpi=(600, 600))

# print(f"✅ Image saved: {output_path} (Optimized for OCR)")
# return output_path


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
#     """Loads and preprocesses an image for OCR."""
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

#     print(f"📏 Image shape: {img.shape}")

#     # 2. Convert to Grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # 3. Apply CLAHE (Contrast Enhancement)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     enhanced_img = clahe.apply(gray)

#     # 4. Apply Adaptive Thresholding
#     _, thresh = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # 5. Save Processed Image
#     cv2.imwrite(output_path, thresh)
#     print(f"✅ Processed image saved: {output_path}")
#     return thresh


# def run_ocr(image_path, output_code_path):
#     """
#     Runs OCR-with-format on the preprocessed image.
#     """
#     print(f"🔍 Running OCR on: {image_path}")

#     try:
#         result = subprocess.run(
#             ["OCR_with_format", image_path, "--method=with_format"],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True,
#         )

#         if result.returncode != 0:
#             print(f"❌ OCR failed: {result.stderr.strip()}")
#             return

#         extracted_code = result.stdout.strip()

#         # Save extracted code
#         with open(output_code_path, "w", encoding="utf-8") as f:
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

#     # Step 1: Preprocess Image
#     processed_image = preprocess_image(image_path, processed_image_path)

#     if processed_image is not None:  # Run OCR only if processing succeeded
#         # Step 2: Run OCR on Processed Image
#         run_ocr(processed_image, OUTPUT_FILE)


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
#     """Loads and preprocesses an image for OCR."""
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

#     print(f"📏 Image shape: {img.shape}")

#     # 2. Convert to Grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # 3. Apply CLAHE (Contrast Enhancement)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     enhanced_img = clahe.apply(gray)

#     # 4. Apply Adaptive Thresholding
#     _, thresh = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # 5. Save Processed Image
#     cv2.imwrite(output_path, thresh)
#     print(f"✅ Processed image saved: {output_path}")

#     return output_path  # Return the path, NOT the NumPy array


# def run_ocr(image_path, output_code_path):
#     """
#     Runs OCR-with-format on the preprocessed image.
#     """
#     print(f"🔍 Running OCR on: {image_path}")

#     try:
#         result = subprocess.run(
#             ["OCR_with_format", image_path, "--method=with_format"],
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

#     # Step 1: Preprocess Image
#     processed_image_path = preprocess_image(image_path, processed_image_path)

#     if processed_image_path is not None:  # ✅ Fixed condition check
#         # Step 2: Run OCR on Processed Image
#         run_ocr(processed_image_path, OUTPUT_FILE)


import os
import subprocess

import cv2
import numpy as np
from PIL import Image

import pytesseract

# Paths
IMAGE_DIR = "output/images"
OUTPUT_FILE = "output/extracted_code.py"
PROCESSED_IMAGE_DIR = "output/processed_images"

# Ensure output directories exist
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)


def preprocess_image(image_path, output_path):
    """Loads and preprocesses an image for OCR."""
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

    print(f"📏 Image shape: {img.shape}")

    # 8. Save Processed Image
    cv2.imwrite(output_path, img)
    print(f"✅ Processed image saved: {output_path}")

    return output_path  # Return the path, NOT the NumPy array


def run_ocr(image_path, output_code_path):
    """
    Runs OCR-with-format on the preprocessed image.
    """
    print(f"🔍 Running OCR on: {image_path}")

    try:

        # Improved Tesseract OCR configuration
        TESSERACT_CONFIG = "--oem 3 --psm 13 -c preserve_interword_spaces=1"

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
