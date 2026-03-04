import cv2
import numpy as np


class DifferentialAnalyzer:
    """
    Computes a binary interaction mask from pixel-wise differences between an
    aligned completed form and its blank reference template.

    User-provided inputs (handwriting, checkbox marks) appear as local
    deviations from the template baseline. Morphological cleanup suppresses
    noise and consolidates regions into a clean input mask.
    """

    def __init__(self, diff_threshold: int = 30, min_region_area: int = 50):
        self.diff_threshold = diff_threshold
        self.min_region_area = min_region_area

    def analyze(
        self, aligned_form: np.ndarray, template: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """
        Produce an interaction mask highlighting user-provided content.

        Parameters
        ----------
        aligned_form : np.ndarray
            Completed form after template registration (grayscale).
        template : np.ndarray
            Blank reference template (grayscale, same dimensions).

        Returns
        -------
        interaction_mask : np.ndarray
            Binary uint8 image; 255 = user input present, 0 = template baseline.
        meta : dict
            - binary_diff      : raw thresholded difference image
            - cleaned_diff     : after morphological open/close
            - diff_region_count: number of valid deviation contours
            - deviation_ratio  : fraction of image area flagged as input
        """
        # ── 1. Absolute pixel-wise difference ─────────────────────────────────
        diff = cv2.absdiff(aligned_form, template)

        _, binary_diff = cv2.threshold(
            diff, self.diff_threshold, 255, cv2.THRESH_BINARY
        )

        # ── 2. Morphological cleanup ───────────────────────────────────────────
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(binary_diff, cv2.MORPH_OPEN, k_open)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)

        # ── 3. Dilate to consolidate nearby deviation regions ─────────────────
        k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dilated = cv2.dilate(cleaned, k_dilate, iterations=2)

        # ── 4. Filter small noise contours ────────────────────────────────────
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        valid = [c for c in contours if cv2.contourArea(c) >= self.min_region_area]

        interaction_mask = np.zeros_like(dilated, dtype=np.uint8)
        for c in valid:
            cv2.drawContours(interaction_mask, [c], -1, 255, cv2.FILLED)

        deviation_ratio = float(np.mean(interaction_mask > 0))

        return interaction_mask, {
            "binary_diff": binary_diff,
            "cleaned_diff": cleaned,
            "diff_region_count": len(valid),
            "deviation_ratio": deviation_ratio,
        }
