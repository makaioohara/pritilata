# Copyright (c) 2025 Pritilata AI Contributors
#
# File Name: main.py
# Description: Entry point for processing test images. Processes all images in the ../data/test directory using the preprocessing pipeline.
# Notes: N/A
# Flow: This file is executed first whenever a user attempts to run a test on a new mammogram image. The input is typically an image in JPG, JPEG, or PNG format. Once the file is provided, it is passed on for pre-processing. Essentially, this file initiates the user experience, allowing the user to try out the outcome of the testing workflow.

import os

from preprocess import preprocess_image
from convert_dicom import convert_dicom_to_png

RAW_IMAGE_DIR = "data/test/user/images/raw"

REQUIRED_IMAGES = ("LMLO", "LCC", "RMLO", "RCC")
IMAGE_FORMATS = {".png", ".jpg", ".jpeg"}
DICOM_FORMAT = ".dcm"


def find_image(name):
    for file in os.listdir(RAW_IMAGE_DIR):
        base, ext = os.path.splitext(file)
        if base == name and ext.lower() in IMAGE_FORMATS | {DICOM_FORMAT}:
            return os.path.join(RAW_IMAGE_DIR, file)
    return None


def prepare_images():
    paths = []

    for name in REQUIRED_IMAGES:
        path = find_image(name)
        if not path:
            print("Required image missing. Preprocessing cancelled.")
            return None

        ext = os.path.splitext(path)[1].lower()
        if ext == DICOM_FORMAT:
            path = convert_dicom_to_png(path)

        paths.append(path)

    return paths


def main():
    image_paths = prepare_images()
    if not image_paths:
        return

    print("Preprocessing images...")
    preprocess_image(image_paths)


if __name__ == "__main__":
    main()
