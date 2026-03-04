import cv2
import numpy as np

from .ocr_extractor import OCRExtractor
from .checkbox_extractor import CheckboxExtractor


class FieldExtractor:
    """
    Routes each form field to the appropriate extractor (OCR or checkbox)
    based on the field type declared in the template definition.

    Field localisation is template-driven: coordinates are consumed from
    predefined field definitions rather than inferred dynamically.
    """

    def __init__(self, tesseract_cmd: str | None = None):
        self.ocr = OCRExtractor(tesseract_cmd=tesseract_cmd)
        self.checkbox = CheckboxExtractor()

    def extract_fields(
        self,
        aligned_form: np.ndarray,
        interaction_mask: np.ndarray,
        field_definitions: list[dict],
    ) -> list[dict]:
        """
        Extract all fields defined in *field_definitions*.

        Parameters
        ----------
        aligned_form : np.ndarray
            Completed form registered to template coordinates (grayscale).
        interaction_mask : np.ndarray
            Binary mask of user-provided input regions from differential
            analysis (same dimensions as *aligned_form*).
        field_definitions : list[dict]
            Each entry must contain:
              id   : str   – unique field identifier
              type : str   – "printed" | "handwritten" | "checkbox"
              x, y : int   – top-left pixel coordinates
              w, h : int   – field width and height in pixels

        Returns
        -------
        list[dict] – one result dict per field with keys:
            field_id, field_type, x, y, w, h,
            value, confidence, raw_data (text) / pixel_density (checkbox)
        """
        results = []

        for fdef in field_definitions:
            fid   = fdef["id"]
            ftype = fdef["type"]
            x, y, w, h = int(fdef["x"]), int(fdef["y"]), int(fdef["w"]), int(fdef["h"])

            # Guard against out-of-bounds coordinates
            img_h, img_w = aligned_form.shape[:2]
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + w, img_w), min(y + h, img_h)

            if x2 <= x1 or y2 <= y1:
                results.append(self._empty_result(fid, ftype, x, y, w, h))
                continue

            roi_form = aligned_form[y1:y2, x1:x2]
            roi_mask = interaction_mask[y1:y2, x1:x2]

            if ftype in ("printed", "handwritten"):
                # Mask the field image to isolate only user-provided content
                masked = cv2.bitwise_and(roi_form, roi_form, mask=roi_mask)
                result = self.ocr.extract(masked, ftype)
            elif ftype == "checkbox":
                result = self.checkbox.extract(roi_form, roi_mask)
            else:
                result = {"value": "", "confidence": 0.0}

            results.append({
                "field_id":   fid,
                "field_type": ftype,
                "x": x, "y": y, "w": w, "h": h,
                **result,
            })

        return results

    @staticmethod
    def _empty_result(fid: str, ftype: str, x: int, y: int, w: int, h: int) -> dict:
        return {
            "field_id":   fid,
            "field_type": ftype,
            "x": x, "y": y, "w": w, "h": h,
            "value": "" if ftype != "checkbox" else False,
            "confidence": 0.0,
        }
