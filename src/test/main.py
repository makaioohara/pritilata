# Copyright (c) 2025 Pritilata AI Contributors
#
# File Name: main.py
# Description: Entry point for processing test images. Processes all images in the ../data/test directory using the preprocessing pipeline.
# Notes: N/A
# Flow: This file is executed first whenever a user attempts to run a test on a new mammogram image. The input is typically an image in JPG, JPEG, or PNG format. Once the file is provided, it is passed on for pre-processing. Essentially, this file initiates the user experience, allowing the user to try out the outcome of the testing workflow.

import os
from preprocess import load_and_preprocess_images
from convert_dicom import convert_dicom_to_png

TESTING_IMAGE_DIR = "data/test/user/images/raw"

IMAGE_VIEWS = ("L-CC", "L-MLO", "R-CC", "R-MLO")
IMAGE_FORMATS = {".png", ".jpg", ".jpeg"}
DICOM_FORMAT = ".dcm"


def find_image(name):
    for file in os.listdir(TESTING_IMAGE_DIR):
        base, ext = os.path.splitext(file)
        if base == name and ext.lower() in IMAGE_FORMATS | {DICOM_FORMAT}:
            return os.path.join(TESTING_IMAGE_DIR, file)
    return None


def prepare_images():
    paths = []

    for name in IMAGE_VIEWS:
        path = find_image(name)
        if not path:
            print("[FAILED] Required image missing. Preprocessing cancelled.")
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
    images_array = load_and_preprocess_images(image_paths)
    print("Preprocessed images shape: ", images_array.shape)

if __name__ == "__main__":
    main()
