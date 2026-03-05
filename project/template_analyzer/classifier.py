"""
Field Classifier
================
Takes raw detected regions from TemplateFieldDetector and:

  1. Assigns a definitive field type (printed | handwritten | checkbox)
  2. Extracts a nearby text label using OCR to generate a meaningful field ID
  3. Infers basic semantic format hints from label keywords

No ML models are used — classification is purely rule-based on geometry,
context, and label keyword matching.
"""

import re
import cv2
import numpy as np

try:
    import pytesseract
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

# ── Label extraction parameters ───────────────────────────────────────────────
_LABEL_SEARCH_LEFT   = 250   # px to the left of region to search for a label
_LABEL_SEARCH_ABOVE  = 40    # px above region to search for a label
_LABEL_SEARCH_RIGHT  = 120   # px to the right (for inline labels)

# ── Keyword-to-format hint mapping ────────────────────────────────────────────
_FORMAT_KEYWORDS: dict[str, list[str]] = {
    "date":         ["date", "dob", "birth", "day", "month", "year", "dd/mm", "mm/dd"],
    "email":        ["email", "e-mail", "mail"],
    "phone":        ["phone", "tel", "mobile", "contact", "fax"],
    "numeric":      ["age", "number", "amount", "score", "count", "total", "no."],
    "alpha":        ["name", "surname", "forename", "first", "last", "title",
                     "gender", "sex", "nationality", "country"],
    "alphanumeric": ["address", "street", "city", "town", "postcode", "zip",
                     "id", "ref", "code", "occupation", "employer"],
}

# ── Area thresholds for type hinting ──────────────────────────────────────────
# Large boxes are more likely to be handwritten; small neat boxes printed
_HW_AREA_THRESHOLD = 15_000   # px² — above this → likely handwritten


class FieldClassifier:
    """
    Classifies detected regions and extracts label metadata.

    Parameters
    ----------
    tesseract_cmd : str | None
        Optional explicit path to the Tesseract binary.
    """

    def __init__(self, tesseract_cmd: str | None = None):
        if tesseract_cmd and _OCR_AVAILABLE:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    # ── Public API ─────────────────────────────────────────────────────────────

    def classify(
        self,
        regions: list[dict],
        template_gray: np.ndarray,
    ) -> list[dict]:
        """
        Classify and label each detected region.

        Parameters
        ----------
        regions       : output of TemplateFieldDetector.detect()
        template_gray : original blank template image (for label extraction)

        Returns
        -------
        list[dict] – enriched region dicts with extra keys:
            field_id        : str   – slugified label or auto-generated ID
            field_type      : str   – "printed" | "handwritten" | "checkbox"
            label_text      : str   – raw OCR label text (may be empty)
            format_hint     : str | None  – inferred semantic format
            required        : bool  – True if label contains "*" or "required"
        """
        used_ids: dict[str, int] = {}
        results   = []

        for idx, region in enumerate(regions):
            raw_type = region["raw_type"]

            # ── 1. Determine field type ────────────────────────────────────────
            if raw_type == "checkbox":
                field_type = "checkbox"
            elif raw_type == "underline":
                field_type = "handwritten"
            else:
                # textbox — use area to guess written vs printed
                field_type = (
                    "handwritten" if region["area"] >= _HW_AREA_THRESHOLD
                    else "printed"
                )

            # ── 2. Extract nearby label ────────────────────────────────────────
            label_text  = self._extract_label(region, template_gray)
            clean_label = self._clean_label(label_text)

            # ── 3. Build a unique field ID ─────────────────────────────────────
            base_id = clean_label if clean_label else f"field_{idx + 1:03d}"
            base_id = self._slugify(base_id)
            field_id = self._unique_id(base_id, used_ids)

            # ── 4. Infer format hint from label keywords ───────────────────────
            format_hint = self._infer_format(label_text) if field_type != "checkbox" else None

            # ── 5. Check for required marker ──────────────────────────────────
            required = "*" in label_text or "required" in label_text.lower()

            results.append({
                **region,
                "field_id":   field_id,
                "field_type": field_type,
                "label_text": label_text.strip(),
                "format_hint": format_hint,
                "required":   required,
            })

        return results

    # ── Label extraction ───────────────────────────────────────────────────────

    def _extract_label(
        self, region: dict, template_gray: np.ndarray
    ) -> str:
        """
        Run OCR on a small area to the left of and above the field region
        to extract the nearest label text.
        """
        if not _OCR_AVAILABLE:
            return ""

        ih, iw = template_gray.shape[:2]
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]

        # Search zone: left side + above the field
        sx1 = max(0, x - _LABEL_SEARCH_LEFT)
        sy1 = max(0, y - _LABEL_SEARCH_ABOVE)
        sx2 = min(iw, x + _LABEL_SEARCH_RIGHT)
        sy2 = min(ih, y + h)

        label_zone = template_gray[sy1:sy2, sx1:sx2]
        if label_zone.size == 0:
            return ""

        # Invert for better OCR on dark-on-white text
        _, zone_bin = cv2.threshold(label_zone, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            text = pytesseract.image_to_string(
                zone_bin,
                config="--oem 3 --psm 6",
            )
        except Exception:
            return ""

        return text.strip()

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_label(text: str) -> str:
        """Strip punctuation / noise characters from raw OCR label."""
        # Keep only meaningful words (≥ 2 chars)
        words = re.findall(r"[A-Za-z][A-Za-z0-9']{1,}", text)
        return " ".join(words[:4])  # take at most 4 words to keep IDs short

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert a label to a lowercase underscore-separated identifier."""
        slug = text.lower().strip()
        slug = re.sub(r"[^a-z0-9]+", "_", slug)
        slug = re.sub(r"_+", "_", slug).strip("_")
        return slug or "field"

    @staticmethod
    def _unique_id(base: str, used: dict[str, int]) -> str:
        """Append a numeric suffix if the base ID has already been used."""
        if base not in used:
            used[base] = 1
            return base
        count = used[base] + 1
        used[base] = count
        return f"{base}_{count}"

    @staticmethod
    def _infer_format(label_text: str) -> str | None:
        """Return a format hint based on keyword presence in the label."""
        lower = label_text.lower()
        for fmt, keywords in _FORMAT_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                return fmt
        return None