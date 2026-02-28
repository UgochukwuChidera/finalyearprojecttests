import cv2
import numpy as np

def binarization(normalized):
    binary = cv2.adaptiveThreshold(
        normalized, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    foreground_ratio = float(np.mean(binary > 0))

    p_fg, p_bg = foreground_ratio, 1 - foreground_ratio
    pixel_entropy = 0.0
    if p_fg > 0: pixel_entropy -= p_fg * np.log2(p_fg)
    if p_bg > 0: pixel_entropy -= p_bg * np.log2(p_bg)

    h, w = binary.shape
    regional = [
        np.mean(binary[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4] > 0)
        for i in range(4) for j in range(4)
    ]

    threshold_stability = float(np.clip(
        1 - np.var(regional) / 0.02, 0, 1
    ))

    return binary, {
        "foreground_ratio": foreground_ratio,
        "pixel_entropy": pixel_entropy,
        "threshold_stability": threshold_stability
    }