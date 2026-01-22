# Copyright (c) 2026 Pritilata AI Contributors
#
# File Name: convert_dicom.py
# Description: Converts a DICOM mammogram image to PNG format, automatically applying rescale slope/intercept. Returns the PNG file path for further processing.
# Notes: NULL
# Flow: This script is intended to be run automatically if there is dicom test input preset during testing.

import numpy as np
import pydicom
import cv2
import png
from pathlib import Path

OUTPUT_DIR = Path("data/test/user/images/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def convert_dicom_to_png(dicom_path, target_size=(896, 1152), output_bitdepth=16):
    dicom_path = Path(dicom_path).resolve()
    png_path = OUTPUT_DIR / f"{dicom_path.stem}.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)

    try:
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

        # Normalize to [0, max_val]
        max_val = (2**output_bitdepth) - 1
        image -= image.min()
        image /= image.max()
        image *= max_val

        # Convert to integer type
        dtype = np.uint16 if output_bitdepth > 8 else np.uint8
        image = image.astype(dtype)

        # Save as PNG
        with open(png_path, "wb") as f:
            writer = png.Writer(
                width=image.shape[1],
                height=image.shape[0],
                bitdepth=output_bitdepth,
                greyscale=True,
            )
            writer.write(f, image.tolist())

        return str(png_path)

    except Exception as e:
        raise RuntimeError(f"Failed to convert!") from e
