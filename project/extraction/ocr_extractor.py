import cv2
import numpy as np
import pytesseract


class OCRExtractor:
    """
    Extracts text from field ROIs using the Tesseract OCR engine.

    Supports both printed and handwritten field types. Returns the
    recognised string value and a normalised confidence score derived
    from Tesseract's per-word confidence output.
    """

    # Tesseract configs: single-line mode with LSTM engine
    _CFG_PRINTED     = "--oem 3 --psm 7"
    _CFG_HANDWRITTEN = "--oem 3 --psm 7"

    def __init__(self, tesseract_cmd: str | None = None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract(self, field_image: np.ndarray, field_type: str = "printed") -> dict:
        """
        Run OCR on *field_image*.

        Returns
        -------
        dict with keys:
          value      : str   – extracted text (stripped)
          confidence : float – mean word confidence in [0, 1]
          raw_data   : dict  – words list and per-word confidence list
        """
        _EMPTY = {"value": "", "confidence": 0.0, "raw_data": {"words": [], "confidences": []}}

        if field_image is None or field_image.size == 0:
            return _EMPTY

        # Ensure the image is uint8 grayscale
        if len(field_image.shape) == 3:
            field_image = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)

        cfg = self._CFG_HANDWRITTEN if field_type == "handwritten" else self._CFG_PRINTED

        try:
            data = pytesseract.image_to_data(
                field_image,
                config=cfg,
                output_type=pytesseract.Output.DICT,
            )
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError(
                "Tesseract is not installed or not on PATH. "
                "Install it and ensure pytesseract.tesseract_cmd is set."
            )

        words, confs = [], []
        for text, conf in zip(data["text"], data["conf"]):
            text = str(text).strip()
            conf = int(conf)
            if text and conf >= 0:
                words.append(text)
                confs.append(conf)

        value = " ".join(words).strip()
        avg_conf = float(np.mean(confs) / 100.0) if confs else 0.0

        return {
            "value": value,
            "confidence": float(np.clip(avg_conf, 0.0, 1.0)),
            "raw_data": {"words": words, "confidences": confs},
        }
