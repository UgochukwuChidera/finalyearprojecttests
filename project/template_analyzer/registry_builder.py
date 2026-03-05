"""
Registry Builder
================
Combines TemplateFieldDetector and FieldClassifier to automatically
generate a complete registry.json entry from a single blank template image.

Usage
-----
    from project.template_analyzer.registry_builder import TemplateRegistryBuilder

    builder = TemplateRegistryBuilder()
    entry   = builder.build(
        template_id  = "template_001",
        image_path   = "templates/template_001.png",
        registry_path = "templates/registry.json",   # written in-place
    )

The produced registry entry is immediately compatible with TemplateRegistry
and the full DAPE pipeline — no manual coordinate entry required.
"""

import json
from pathlib import Path

import cv2

from .detector   import TemplateFieldDetector
from .classifier import FieldClassifier


class TemplateRegistryBuilder:
    """
    Automatically generates template registry entries from blank form images.

    Parameters
    ----------
    tesseract_cmd      : optional path to Tesseract binary
    checkbox_max_area  : override for checkbox area threshold
    textbox_min_area   : override for text-box area threshold
    line_min_width     : override for underline detection threshold
    padding            : pixel padding added around detected regions
    """

    def __init__(
        self,
        tesseract_cmd:     str | None = None,
        checkbox_max_area: int        = 3000,
        textbox_min_area:  int        = 3000,
        line_min_width:    int        = 60,
        padding:           int        = 4,
    ):
        self._detector   = TemplateFieldDetector(
            checkbox_max_area = checkbox_max_area,
            textbox_min_area  = textbox_min_area,
            line_min_width    = line_min_width,
            padding           = padding,
        )
        self._classifier = FieldClassifier(tesseract_cmd=tesseract_cmd)

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(
        self,
        template_id:   str,
        image_path:    str,
        registry_path: str  = "templates/registry.json",
        save:          bool = True,
    ) -> dict:
        """
        Analyse a blank template image and produce a registry entry.

        Parameters
        ----------
        template_id   : unique string key for this template
        image_path    : path to the blank template image file
        registry_path : where to read/write the registry JSON
        save          : if True, persist the entry to registry_path

        Returns
        -------
        dict – the registry entry that was (or would be) saved:
            {
              "image_path":    str,
              "fields":        list[dict],
              "output_schema": dict[str, str]
            }
        """
        # ── Load template image ────────────────────────────────────────────────
        image_path = str(image_path)
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Cannot load template image: {image_path!r}")

        # ── Detect and classify fields ─────────────────────────────────────────
        raw_regions  = self._detector.detect(gray)
        classified   = self._classifier.classify(raw_regions, gray)

        # ── Build field definitions compatible with TemplateRegistry ───────────
        fields = []
        for region in classified:
            field_def: dict = {
                "id":   region["field_id"],
                "type": region["field_type"],
                "x":    region["x"],
                "y":    region["y"],
                "w":    region["w"],
                "h":    region["h"],
            }
            if region.get("required"):
                field_def["required"] = True
            if region.get("format_hint"):
                field_def["format"] = region["format_hint"]

            fields.append(field_def)

        # ── Build output schema (field_id → semantic_key) ─────────────────────
        output_schema = {f["id"]: f["id"] for f in fields}

        entry = {
            "image_path":    image_path,
            "fields":        fields,
            "output_schema": output_schema,
        }

        # ── Persist to registry ────────────────────────────────────────────────
        if save:
            self._write_to_registry(template_id, entry, registry_path)

        print(
            f"[RegistryBuilder] Template '{template_id}' → "
            f"{len(fields)} fields detected "
            f"({sum(1 for f in fields if f['type']=='checkbox')} checkboxes, "
            f"{sum(1 for f in fields if f['type']=='handwritten')} handwritten, "
            f"{sum(1 for f in fields if f['type']=='printed')} printed)"
        )

        return entry

    def build_batch(
        self,
        templates: list[dict],
        registry_path: str = "templates/registry.json",
    ) -> dict[str, dict]:
        """
        Register multiple templates at once.

        Parameters
        ----------
        templates : list of dicts, each with keys:
            template_id : str
            image_path  : str
        registry_path : shared registry file

        Returns
        -------
        dict mapping template_id → generated entry
        """
        results = {}
        for t in templates:
            entry = self.build(
                template_id   = t["template_id"],
                image_path    = t["image_path"],
                registry_path = registry_path,
                save          = True,
            )
            results[t["template_id"]] = entry
        return results

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _write_to_registry(
        template_id: str,
        entry:       dict,
        registry_path: str,
    ) -> None:
        """Load existing registry, upsert this entry, and save."""
        path = Path(registry_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        registry: dict = {}
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as fh:
                    registry = json.load(fh)
            except json.JSONDecodeError:
                registry = {}

        registry[template_id] = entry

        with path.open("w", encoding="utf-8") as fh:
            json.dump(registry, fh, indent=2, ensure_ascii=False)

        print(f"[RegistryBuilder] Registry saved → {registry_path}")