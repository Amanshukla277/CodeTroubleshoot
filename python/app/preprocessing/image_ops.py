import cv2
import numpy as np


def decode_image(data):
    array = np.frombuffer(data, dtype=np.uint8).copy()
    image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Invalid image")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        # Transparent PNG: put it on a white background instead of black
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        color = image[:, :, :3].astype(np.float32)
        image = (color * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return image


def resize_image(image, max_width=800, max_height=600):
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return image
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def to_grayscale(image):
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_ink_channel(image):
    # Darkest channel: coloured text (yellow, orange, cyan) becomes dark on a white background
    if image.ndim == 2:
        return image
    return image.min(axis=2)


def preprocess_image(image):
    gray = to_ink_channel(image)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary