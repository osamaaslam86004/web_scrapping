import os

from paddleocr import PaddleOCR

# Path to images and output file
IMAGE_DIR = "output/images"
OUTPUT_FILE = "output/extracted_code.py"


for image_file in os.listdir(IMAGE_DIR):
    image_path = os.path.join(IMAGE_DIR, image_file)

    ocr = PaddleOCR(
        use_angle_cls=True, lang="en"
    )  # need to run only once to download and load model into memory
    result = ocr.ocr("path_to_image.png", cls=True)

    for line in result:
        print(line)
