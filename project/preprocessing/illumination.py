import cv2
import numpy as np

def illumination_normalization(gray, grayscale_std):
    background = cv2.GaussianBlur(gray, (51, 51), 0)
    corrected = cv2.divide(gray, background + 1, scale=255)

    clahe = cv2.createCLAHE(2.0, (8, 8))
    normalized = clahe.apply(corrected)

    illumination_std = float(np.std(background))
    illumination_uniformity = float(np.clip(
        1 - illumination_std / 60, 0, 1
    ))

    local_contrast_gain = float(np.clip(
        (np.std(normalized) / (grayscale_std + 1e-5)) / 1.5, 0, 1
    ))

    return normalized, {
        "illumination_uniformity": illumination_uniformity,
        "local_contrast_gain": local_contrast_gain
    }