"""
Template Registry
=================
Manages the collection of blank form templates used by the DAPE pipeline.

Each template entry in ``templates/registry.json`` must contain:

  image_path    : str         – path to the blank template image file
  fields        : list[dict]  – field definitions; each entry must have:
      id            : str           – unique field identifier
      type          : str           – "printed" | "handwritten" | "checkbox"
      x, y, w, h    : int           – bounding box in template pixel coords
      required      : bool          (optional, default False)
      format        : str           (optional) – semantic format key
      max_length    : int           (optional)
      allowed_values: list[str]     (optional)
  output_schema : dict[str,str] – maps field_id → semantic_key for output

Example registry.json
---------------------
{
  "template_001": {
    "image_path": "templates/template_001.png",
    "fields": [
      {"id": "name",  "type": "handwritten", "x": 120, "y": 210, "w": 320, "h": 45,
       "required": true, "format": "alpha"},
      {"id": "dob",   "type": "printed",     "x": 120, "y": 280, "w": 200, "h": 40,
       "format": "date"},
      {"id": "agree", "type": "checkbox",    "x":  55, "y": 370, "w":  30, "h": 30}
    ],
    "output_schema": {
      "name":  "respondent_name",
      "dob":   "date_of_birth",
      "agree": "consent_given"
    }
  }
}
"""

import json
from pathlib import Path

import cv2
import numpy as np


class TemplateRegistry:
    """
    Loads and provides access to form template definitions.

    Templates are registered via a JSON file (default:
    ``templates/registry.json``).  New templates can also be registered
    programmatically and persisted back to that file.
    """

    def __init__(self, registry_path: str = "templates/registry.json"):
        self._path = Path(registry_path)
        self._data: dict = {}
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            with self._path.open("r", encoding="utf-8") as fh:
                self._data = json.load(fh)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, ensure_ascii=False)

    # ── Accessors ──────────────────────────────────────────────────────────────

    def list_templates(self) -> list[str]:
        return list(self._data.keys())

    def get_entry(self, template_id: str) -> dict:
        if template_id not in self._data:
            raise KeyError(
                f"Template '{template_id}' not found in registry. "
                f"Available: {self.list_templates()}"
            )
        return self._data[template_id]

    def get_template_image(self, template_id: str) -> np.ndarray:
        """Load and return the blank template as a grayscale ndarray."""
        entry      = self.get_entry(template_id)
        image_path = entry["image_path"]
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(
                f"Template image not found: '{image_path}' "
                f"(registered under '{template_id}')"
            )
        return img

    def get_field_definitions(self, template_id: str) -> list[dict]:
        return self.get_entry(template_id).get("fields", [])

    def get_output_schema(self, template_id: str) -> dict[str, str]:
        return self.get_entry(template_id).get("output_schema", {})

    # ── Registration ───────────────────────────────────────────────────────────

    def register_template(
        self,
        template_id: str,
        image_path: str,
        fields: list[dict],
        output_schema: dict[str, str] | None = None,
    ) -> None:
        """
        Add or overwrite a template entry and persist the registry.

        If *output_schema* is omitted each field_id is used as its own
        semantic key.
        """
        self._data[template_id] = {
            "image_path":    image_path,
            "fields":        fields,
            "output_schema": output_schema or {f["id"]: f["id"] for f in fields},
        }
        self._save()
