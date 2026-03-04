import numpy as np


class CheckboxExtractor:
    """
    Classifies a checkbox field as marked or unmarked using pixel density
    within the differential interaction mask for that field region.

    Confidence is derived from the magnitude of the pixel density relative
    to the decision boundary, yielding low confidence in ambiguous cases.
    """

    def __init__(
        self,
        marked_threshold: float = 0.15,
        unmarked_threshold: float = 0.04,
    ):
        self.marked_threshold = marked_threshold
        self.unmarked_threshold = unmarked_threshold

    def extract(
        self,
        field_image: np.ndarray,
        diff_roi: np.ndarray | None = None,
    ) -> dict:
        """
        Parameters
        ----------
        field_image : np.ndarray
            Cropped ROI from the aligned completed form (grayscale uint8).
        diff_roi : np.ndarray or None
            Corresponding region from the interaction mask. If provided,
            pixel density is measured on the mask (preferred); otherwise
            the field image itself is used.

        Returns
        -------
        dict with keys:
          value         : bool  – True if marked
          confidence    : float – in [0, 1]
          pixel_density : float – fraction of non-zero pixels in region
        """
        _EMPTY = {"value": False, "confidence": 0.0, "pixel_density": 0.0}

        region = diff_roi if diff_roi is not None else field_image
        if region is None or region.size == 0:
            return _EMPTY

        pixel_density = float(np.count_nonzero(region) / region.size)

        mid = (self.marked_threshold + self.unmarked_threshold) / 2.0

        if pixel_density >= self.marked_threshold:
            value = True
            conf_range = max(0.5 - self.marked_threshold, 1e-6)
            confidence = float(
                np.clip((pixel_density - self.marked_threshold) / conf_range, 0.0, 1.0)
            )
        elif pixel_density <= self.unmarked_threshold:
            value = False
            conf_range = max(self.unmarked_threshold, 1e-6)
            confidence = float(
                np.clip(1.0 - pixel_density / conf_range, 0.0, 1.0)
            )
        else:
            # Ambiguous zone – assign low confidence
            value = pixel_density > mid
            confidence = 0.25

        return {
            "value": value,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "pixel_density": pixel_density,
        }
