# Copyright (c) 2026 Pritilata AI Contributors
#
# File Name: convert_dicom.py
# Description: Converts DICOM mammogram images to PNG format with optional resizing and normalization. Supports 8-bit or 16-bit PNG output and applies DICOM rescale slope/intercept automatically.
# Notes: Input and output paths are resolved relative to the project root.
# Flow: This script is intended to be run manually after dataset download and before metadata generation.

import numpy as np
import pydicom
import cv2
import png
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

DICOM_ROOT = BASE_DIR / "data/train/ddsm/images/dicom"
PNG_ROOT = BASE_DIR / "data/train/ddsm/images/png"
IMAGE_ROOT = BASE_DIR / "data/train/ddsm/images"

MAX_FOLDER_SIZE_GB = 180


def get_folder_size_gb(path: Path) -> float:
    total_bytes = 0
    for file in path.rglob("*"):
        if file.is_file():
            total_bytes += file.stat().st_size
    return total_bytes / (1024 ** 3)


def save_dicom_image_as_png(
    dicom_path: Path,
    png_path: Path,
    target_size=(896, 1152),
    output_bitdepth=16,
):
    png_path.parent.mkdir(parents=True, exist_ok=True)

    ds = pydicom.dcmread(dicom_path)
    image = ds.pixel_array.astype(np.float32)

    # Apply DICOM rescale if present
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    image = image * slope + intercept

    # Resize if needed
    if target_size is not None:
        image = cv2.resize(
            image,
            dsize=target_size,
            interpolation=cv2.INTER_CUBIC,
        )

    # Normalize
    max_val = (2 ** output_bitdepth) - 1
    image -= image.min()
    image /= image.max()
    image *= max_val
    image = image.astype(np.uint16 if output_bitdepth > 8 else np.uint8)

    # Save PNG
    with open(png_path, "wb") as f:
        writer = png.Writer(
            width=image.shape[1],
            height=image.shape[0],
            bitdepth=output_bitdepth,
            greyscale=True,
        )
        writer.write(f, image.tolist())


def convert_all_dicoms():
    for dicom_file in DICOM_ROOT.rglob("*.dcm"):

        # Check folder size before each conversion
        current_size = get_folder_size_gb(IMAGE_ROOT)
        if current_size > MAX_FOLDER_SIZE_GB:
            print(
                f"[STOPPED] Image folder size exceeded!"
            )
            return

        # Preserve folder structure
        relative_path = dicom_file.relative_to(DICOM_ROOT)
        png_path = PNG_ROOT / relative_path.with_suffix(".png")

        try:
            save_dicom_image_as_png(dicom_file, png_path)
        except Exception as e:
            print(f"[FAILED] {dicom_file}: {e}")


if __name__ == "__main__":
    convert_all_dicoms()
