import cv2
import numpy as np

def structure_prep(binary, template_size):
    ch, cw = binary.shape
    tw, th = template_size

    scale_consistency = float(np.clip(
        1 - abs(tw / cw - th / ch) / 0.1, 0, 1
    ))

    orb = cv2.ORB_create(500)
    keypoints = orb.detect(binary, None)

    feature_count = len(keypoints)
    structural_confidence = float(np.clip(
        0.4 * scale_consistency +
        0.6 * min(feature_count / 500, 1), 0, 1
    ))

    return {
        "feature_count": feature_count,
        "structural_confidence": structural_confidence
    }