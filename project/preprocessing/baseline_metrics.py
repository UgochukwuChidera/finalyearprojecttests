import cv2
import numpy as np

def baseline_metrics(gray):
    grayscale_mean = float(np.mean(gray))
    grayscale_std  = float(np.std(gray))
    min_intensity  = int(np.min(gray))
    max_intensity  = int(np.max(gray))
    intensity_range = float(max_intensity - min_intensity)

    near_white_ratio = float(np.mean(gray > 240))
    near_black_ratio = float(np.mean(gray < 15))

    contrast_score = float(np.clip((intensity_range - 80) / 120, 0, 1))

    white_score = np.clip(1 - abs(near_white_ratio - 0.6), 0, 1)
    black_score = np.clip(1 - near_black_ratio / 0.15, 0, 1)
    balance_score = float(0.6 * white_score + 0.4 * black_score)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = float(np.clip((lap_var - 50) / 150, 0, 1))

    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    noise_level = float(np.mean(cv2.absdiff(gray, denoised)))
    noise_score = float(np.clip(1 - noise_level / 40, 0, 1))

    return {
        "grayscale_mean": grayscale_mean,
        "grayscale_std": grayscale_std,
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,
        "intensity_range": intensity_range,
        "near_white_ratio": near_white_ratio,
        "near_black_ratio": near_black_ratio,
        "contrast_score": contrast_score,
        "balance_score": balance_score,
        "blur_score": blur_score,
        "noise_score": noise_score,
    }