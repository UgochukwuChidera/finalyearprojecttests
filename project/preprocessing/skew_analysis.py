import cv2
import numpy as np

def skew_analysis(gray):
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 10
    )

    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    skew_angle = 0.0
    skew_confidence = 0.0
    skew_acceptability = 1.0

    if lines is not None:
        angles = np.array([
            (theta - np.pi / 2) * 180 / np.pi
            for rho, theta in lines[:30, 0]
        ])

        median = np.median(angles)
        mean = np.mean(angles)
        skew_angle = float(0.8 * median + 0.2 * mean)

        conf_std = 1 - np.std(angles) / 2.0
        conf_agree = 1 - abs(mean - median) / 2.0
        skew_confidence = float(np.clip(
            0.7 * conf_std + 0.3 * conf_agree, 0, 1
        ))

        skew_acceptability = float(np.clip(
            1 - abs(skew_angle) / 3.0, 0, 1
        ))

    skew_risk = float(np.clip(abs(skew_angle) / 5.0, 0, 1))
    effective_skew_risk = float(skew_confidence * skew_risk)

    return {
        "skew_angle": skew_angle,
        "skew_confidence": skew_confidence,
        "skew_acceptability": skew_acceptability,
        "skew_risk": skew_risk,
        "effective_skew_risk": effective_skew_risk
    }