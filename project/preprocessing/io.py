import cv2

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Image could not be loaded.")
    h, w = image.shape[:2]
    aspect_ratio = w / h
    return image, h, w, aspect_ratio