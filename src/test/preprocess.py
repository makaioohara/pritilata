import numpy as np
from PIL import Image

def load_and_preprocess_images(image_paths):
    processed_images = []

    for path in image_paths:
        image = load_image(path)
        processed_images.append(image)

    # Stack all images into one array: (batch, H, W, 1)
    return np.vstack(processed_images)


def load_image(image_path):
    # Load image using PIL
    img = Image.open(image_path)

    # Convert to grayscale if not already
    if img.mode != 'L':
        img = img.convert('L')

    # Convert to numpy array
    image = np.array(img, dtype=np.float32)

    # Normalize
    normalize_image(image)

    # Add batch dimension and channel dimension
    image = np.expand_dims(image, axis=0)   # batch
    image = np.expand_dims(image, axis=-1)  # channel

    return image


def normalize_image(image):
    image -= np.mean(image)
    image /= np.std(image)
