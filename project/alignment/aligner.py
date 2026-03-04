import cv2
import numpy as np


class TemplateAligner:
    """
    Aligns a scanned completed form to its blank reference template using
    ORB feature matching and perspective (homography) transformation.

    Corrects translation, rotation, and scale differences introduced during
    scanning, establishing pixel-level correspondence required by the
    differential analysis stage.
    """

    def __init__(self, max_features: int = 1000, match_ratio: float = 0.75):
        self.max_features = max_features
        self.match_ratio = match_ratio

    def align(
        self, scanned: np.ndarray, template: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Align *scanned* (completed form) to *template* (blank form).

        Returns
        -------
        aligned : np.ndarray
            Warped version of *scanned* in the coordinate space of *template*.
        transform : np.ndarray
            3×3 homography matrix used for the warp.
        meta : dict
            Alignment quality indicators:
              - alignment_confidence  [0, 1]
              - inlier_count          int
              - match_count           int
        """
        _NULL = (
            scanned.copy(),
            np.eye(3, dtype=np.float64),
            {"alignment_confidence": 0.0, "inlier_count": 0, "match_count": 0},
        )

        orb = cv2.ORB_create(self.max_features)
        kp1, des1 = orb.detectAndCompute(scanned, None)
        kp2, des2 = orb.detectAndCompute(template, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return _NULL

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = matcher.knnMatch(des1, des2, k=2)

        good = [
            m for m, n in raw_matches if m.distance < self.match_ratio * n.distance
        ]

        if len(good) < 4:
            return (
                scanned.copy(),
                np.eye(3, dtype=np.float64),
                {"alignment_confidence": 0.0, "inlier_count": 0, "match_count": len(good)},
            )

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is None:
            return (
                scanned.copy(),
                np.eye(3, dtype=np.float64),
                {"alignment_confidence": 0.0, "inlier_count": 0, "match_count": len(good)},
            )

        inliers = int(mask.sum()) if mask is not None else 0
        h, w = template.shape[:2]
        aligned = cv2.warpPerspective(scanned, M, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)

        confidence = float(np.clip(inliers / max(len(good), 1), 0.0, 1.0))

        return aligned, M, {
            "alignment_confidence": confidence,
            "inlier_count": inliers,
            "match_count": len(good),
        }
