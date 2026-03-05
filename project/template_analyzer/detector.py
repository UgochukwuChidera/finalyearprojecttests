"""
Template Field Detector
=======================
Analyses a blank form image using computer vision to automatically detect
candidate field regions. Finds:

  - Bordered rectangles / boxes      → text fields (printed or handwritten)
  - Underscored / single-line areas  → handwritten input lines
  - Small square regions             → checkboxes

No machine learning is involved. Detection uses morphological operations,
contour analysis, and Hough line transforms only.
"""

import cv2
import numpy as np


# ── Tuneable detection parameters ─────────────────────────────────────────────

# Checkbox: square-ish regions smaller than this area (px²)
_CHECKBOX_MAX_AREA    = 3000
_CHECKBOX_MIN_AREA    = 100
_CHECKBOX_MAX_ASPECT  = 1.6   # width/height ratio tolerance

# Text field boxes: bordered rectangles above this area
_TEXTBOX_MIN_AREA     = 3000
_TEXTBOX_MIN_WIDTH    = 60
_TEXTBOX_MIN_HEIGHT   = 15

# Underline fields: horizontal line segments
_LINE_MIN_WIDTH       = 60    # minimum pixel length to count as an input line
_LINE_MAX_HEIGHT      = 8     # maximum vertical thickness of an underline

# Duplicate suppression: merge regions closer than this many pixels
_MERGE_DISTANCE       = 12


class TemplateFieldDetector:
    """
    Detects field regions in a blank template image.

    Parameters
    ----------
    checkbox_max_area : int
        Maximum bounding-box area (px²) for a region to be considered a checkbox.
    textbox_min_area : int
        Minimum bounding-box area for a bordered rectangle to be a text field.
    line_min_width : int
        Minimum width for a horizontal segment to be treated as an input line.
    padding : int
        Pixels added around each detected region boundary (guards against
        tight crops that cut off ascenders / descenders).
    """

    def __init__(
        self,
        checkbox_max_area: int = _CHECKBOX_MAX_AREA,
        textbox_min_area:  int = _TEXTBOX_MIN_AREA,
        line_min_width:    int = _LINE_MIN_WIDTH,
        padding:           int = 4,
    ):
        self.checkbox_max_area = checkbox_max_area
        self.textbox_min_area  = textbox_min_area
        self.line_min_width    = line_min_width
        self.padding           = padding

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(self, template_gray: np.ndarray) -> list[dict]:
        """
        Detect all candidate field regions in *template_gray*.

        Parameters
        ----------
        template_gray : np.ndarray
            Grayscale blank template image (uint8).

        Returns
        -------
        list[dict] – detected regions, each with keys:
            x, y, w, h      : int  – bounding box in template pixel coords
            raw_type        : str  – "checkbox" | "textbox" | "underline"
            aspect_ratio    : float
            area            : int
        Sorted top-to-bottom, left-to-right (reading order).
        """
        binary = self._binarize(template_gray)

        checkboxes  = self._detect_checkboxes(binary)
        textboxes   = self._detect_textboxes(binary, template_gray)
        underlines  = self._detect_underlines(binary)

        all_regions = checkboxes + textboxes + underlines
        all_regions = self._suppress_duplicates(all_regions)
        all_regions = self._apply_padding(all_regions, template_gray.shape)
        all_regions.sort(key=lambda r: (r["y"] // 20, r["x"]))   # reading order

        return all_regions

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        """Produce a clean binary image highlighting structural lines."""
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary  = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15, 4
        )
        # Light denoise to remove specks without destroying thin lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return binary

    # ── Checkbox detection ─────────────────────────────────────────────────────

    def _detect_checkboxes(self, binary: np.ndarray) -> list[dict]:
        """Find small, roughly square bordered regions."""
        # Emphasise closed rectangular shapes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        results = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area   = w * h
            aspect = w / max(h, 1)

            if (
                _CHECKBOX_MIN_AREA <= area <= self.checkbox_max_area
                and _CHECKBOX_MAX_ASPECT >= aspect >= 1 / _CHECKBOX_MAX_ASPECT
                and w >= 10 and h >= 10
            ):
                results.append({
                    "x": x, "y": y, "w": w, "h": h,
                    "raw_type": "checkbox",
                    "aspect_ratio": round(aspect, 3),
                    "area": area,
                })

        return results

    # ── Bordered text-box detection ────────────────────────────────────────────

    def _detect_textboxes(
        self, binary: np.ndarray, gray: np.ndarray
    ) -> list[dict]:
        """Find larger bordered rectangles (printed / handwritten text fields)."""
        # Extract horizontal and vertical lines separately to find closed boxes
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))

        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        # Combine to get only box-forming structure
        combined = cv2.bitwise_or(h_lines, v_lines)
        close_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed   = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_k)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        ih, iw = gray.shape[:2]
        results = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area   = w * h
            aspect = w / max(h, 1)

            # Skip full-page borders and tiny noise
            if area > 0.85 * ih * iw:
                continue
            if (
                area >= self.textbox_min_area
                and w >= _TEXTBOX_MIN_WIDTH
                and h >= _TEXTBOX_MIN_HEIGHT
            ):
                results.append({
                    "x": x, "y": y, "w": w, "h": h,
                    "raw_type": "textbox",
                    "aspect_ratio": round(aspect, 3),
                    "area": area,
                })

        return results

    # ── Underline field detection ──────────────────────────────────────────────

    def _detect_underlines(self, binary: np.ndarray) -> list[dict]:
        """
        Detect standalone horizontal lines that serve as write-on underlines.
        These are typically 1-5 px tall but can be quite wide.
        """
        h_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.line_min_width, 1)
        )
        h_only = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        contours, _ = cv2.findContours(
            h_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        results = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= self.line_min_width and h <= _LINE_MAX_HEIGHT:
                # Expand the bounding box upward to capture writing above the line
                expansion = max(30, int(w * 0.08))
                y_new = max(0, y - expansion)
                h_new = h + expansion

                results.append({
                    "x": x, "y": y_new, "w": w, "h": h_new,
                    "raw_type": "underline",
                    "aspect_ratio": round(w / max(h_new, 1), 3),
                    "area": w * h_new,
                })

        return results

    # ── Post-processing ────────────────────────────────────────────────────────

    def _suppress_duplicates(self, regions: list[dict]) -> list[dict]:
        """
        Remove near-duplicate detections (same area detected by multiple passes).
        Uses simple IoU-like overlap check.
        """
        if not regions:
            return []

        kept = []
        for r in regions:
            duplicate = False
            for k in kept:
                overlap = self._overlap(r, k)
                if overlap > 0.5:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(r)
        return kept

    @staticmethod
    def _overlap(a: dict, b: dict) -> float:
        """Intersection-over-union for two bounding boxes."""
        ax1, ay1 = a["x"], a["y"]
        ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
        bx1, by1 = b["x"], b["y"]
        bx2, by2 = bx1 + b["w"], by1 + b["h"]

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter    = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union    = a["w"] * a["h"] + b["w"] * b["h"] - inter
        return inter / max(union, 1)

    def _apply_padding(
        self, regions: list[dict], shape: tuple[int, int]
    ) -> list[dict]:
        """Add small padding around each region, clamped to image bounds."""
        ih, iw = shape
        p = self.padding
        padded = []
        for r in regions:
            padded.append({
                **r,
                "x": max(0, r["x"] - p),
                "y": max(0, r["y"] - p),
                "w": min(iw - r["x"] + p, r["w"] + 2 * p),
                "h": min(ih - r["y"] + p, r["h"] + 2 * p),
            })
        return padded