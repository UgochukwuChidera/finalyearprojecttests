"""
DAPE Pipeline Orchestrator
==========================
Executes the full Differential Analysis and Preprocessing for Extraction
(DAPE) pipeline as a strictly sequential, module-driven workflow:

  1. Preprocessing       - normalize scanned input image
  2. Template Alignment  - register completed form to blank template
  3. Differential Analysis - isolate user-provided input regions
  4. Field Extraction    - OCR / checkbox detection within masked regions
  5. Confidence Validation - flag uncertain / semantically invalid fields
  6. HITL Escalation     - optional human correction of flagged fields
  7. Output Structuring  - map to semantic keys → canonical JSON
  8. Export & Audit Log  - write JSON/CSV/XLSX and per-form audit record

Architectural note
------------------
Each stage consumes explicit outputs from the preceding stage only.
No hidden state is shared between modules. System parameters (thresholds,
schema, template definitions) are passed in at construction time and are
never modified by human corrections or pipeline outputs.
"""

from pathlib import Path

# ── Preprocessing ──────────────────────────────────────────────────────────────
from .preprocessing.io               import load_image
from .preprocessing.grayscale        import to_grayscale
from .preprocessing.baseline_metrics import baseline_metrics
from .preprocessing.skew_analysis    import skew_analysis
from .preprocessing.illumination     import illumination_normalization
from .preprocessing.binarization     import binarization
from .preprocessing.border_removal   import border_removal
from .preprocessing.structure_prep   import structure_prep
from .preprocessing.fusion           import fuse

# ── Remaining pipeline stages ─────────────────────────────────────────────────
from .template_registry               import TemplateRegistry
from .alignment.aligner               import TemplateAligner
from .differential.analyzer           import DifferentialAnalyzer
from .extraction.field_extractor      import FieldExtractor
from .validation.confidence_validator import ConfidenceValidator
from .hitl.escalation                 import HITLEscalation
from .hitl.interface                  import HITLInterface
from .output.structurer               import OutputStructurer
from .output.exporter                 import DataExporter
from .output.audit_logger             import AuditLogger


class DAPEOrchestrator:
    """
    End-to-end DAPE pipeline controller.

    Parameters
    ----------
    registry_path       : path to templates/registry.json
    output_dir          : directory for exported form outputs
    log_dir             : directory for audit logs
    confidence_threshold: fields below this score are flagged for HITL
    enable_hitl         : set False to skip human review
    hitl_host / port    : address for the Flask review interface
    tesseract_cmd       : optional explicit path to the Tesseract binary
    """

    def __init__(
        self,
        registry_path:        str        = "templates/registry.json",
        output_dir:           str        = "outputs",
        log_dir:              str        = "logs",
        confidence_threshold: float      = 0.60,
        enable_hitl:          bool       = True,
        hitl_host:            str        = "127.0.0.1",
        hitl_port:            int        = 5050,
        tesseract_cmd:        str | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._registry   = TemplateRegistry(registry_path)
        self._aligner    = TemplateAligner()
        self._differ     = DifferentialAnalyzer()
        self._extractor  = FieldExtractor(tesseract_cmd=tesseract_cmd)
        self._validator  = ConfidenceValidator(confidence_threshold)
        self._escalation = HITLEscalation()
        self._exporter   = DataExporter()
        self._logger     = AuditLogger(log_dir)

        self._enable_hitl = enable_hitl
        self._hitl_ui: HITLInterface | None = (
            HITLInterface(hitl_host, hitl_port) if enable_hitl else None
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(
        self,
        image_path:  str,
        template_id: str,
        form_id:     str | None = None,
    ) -> dict:
        """
        Process a single scanned completed form through the full pipeline.

        Returns
        -------
        dict with keys:
          structured_output : canonical JSON-compatible document
          export_paths      : {json, csv, xlsx} file locations
          audit_log_path    : path to the per-form audit log
          stats             : all pipeline metrics
          images            : intermediate image dict
        """
        form_id = form_id or Path(image_path).stem
        stats:  dict = {}
        images: dict = {}

        # ── Stage 1 · Preprocessing ────────────────────────────────────────────
        image, h, w, aspect_ratio = load_image(image_path)
        gray = to_grayscale(image)

        stats.update({"original_width": w, "original_height": h, "aspect_ratio": aspect_ratio})
        stats.update(baseline_metrics(gray))
        stats.update(skew_analysis(gray))

        normalized, illum = illumination_normalization(gray, stats["grayscale_std"])
        stats.update(illum)

        binary, bin_stats = binarization(normalized)
        stats.update(bin_stats)

        cropped, crop_stats = border_removal(binary, stats["threshold_stability"])
        stats.update(crop_stats)

        template_size = self._get_template_size(template_id)
        stats.update(structure_prep(cropped, template_size))
        stats["fusion_score"] = fuse(stats)

        images.update({
            "gray":           gray,
            "normalized":     normalized,
            "binary":         binary,
            "cropped_binary": cropped,
        })

        # ── Stage 2 · Template Alignment ──────────────────────────────────────
        template_img  = self._registry.get_template_image(template_id)
        field_defs    = self._registry.get_field_definitions(template_id)
        output_schema = self._registry.get_output_schema(template_id)

        aligned, transform, align_meta = self._aligner.align(gray, template_img)
        stats.update({f"align_{k}": v for k, v in align_meta.items()})
        images["aligned"] = aligned

        # ── Stage 3 · Differential Analysis ───────────────────────────────────
        interaction_mask, diff_meta = self._differ.analyze(aligned, template_img)
        stats.update({
            f"diff_{k}": v for k, v in diff_meta.items()
            if not hasattr(v, "shape")   # exclude ndarray values
        })
        images["interaction_mask"] = interaction_mask
        images["binary_diff"]      = diff_meta["binary_diff"]

        # ── Stage 4 · Field Extraction ─────────────────────────────────────────
        extracted_fields = self._extractor.extract_fields(
            aligned, interaction_mask, field_defs
        )

        # ── Stage 5 · Confidence-Based Validation ─────────────────────────────
        validated_fields = self._validator.validate(extracted_fields, field_defs)

        # ── Stage 6 · Human-in-the-Loop Escalation ────────────────────────────
        flagged = self._escalation.get_flagged(validated_fields)
        stats["hitl_flagged_count"] = len(flagged)

        if flagged and self._enable_hitl and self._hitl_ui is not None:
            corrections  = self._hitl_ui.run_review(flagged)
            validated_fields = self._escalation.apply_bulk_corrections(
                corrections, validated_fields
            )

        esc_stats = self._escalation.escalation_stats(validated_fields)
        stats.update({f"esc_{k}": v for k, v in esc_stats.items()})

        # ── Stage 7 · Output Structuring ──────────────────────────────────────
        structurer        = OutputStructurer(output_schema)
        structured_output = structurer.structure(
            validated_fields, form_id, template_id, stats
        )

        # ── Stage 8 · Export & Audit Logging ──────────────────────────────────
        base_path      = str(self.output_dir / form_id)
        export_paths   = self._exporter.export_all(structured_output, base_path)
        audit_log_path = self._logger.log(
            form_id, template_id, stats, validated_fields, export_paths
        )

        return {
            "structured_output": structured_output,
            "export_paths":      export_paths,
            "audit_log_path":    audit_log_path,
            "stats":             stats,
            "images":            images,
        }

    def process_batch(
        self,
        image_paths: list[str],
        template_id: str,
    ) -> list[dict]:
        """
        Process a list of scanned forms against the same template.

        Errors for individual forms are caught and recorded rather than
        aborting the batch.
        """
        results = []
        for path in image_paths:
            form_id = Path(path).stem
            try:
                result = self.process(path, template_id, form_id)
            except Exception as exc:  # noqa: BLE001
                result = {
                    "structured_output": None,
                    "export_paths":      {},
                    "audit_log_path":    None,
                    "stats":             {"error": str(exc)},
                    "images":            {},
                    "form_id":           form_id,
                    "error":             str(exc),
                }
            results.append(result)
        return results

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_template_size(self, template_id: str) -> tuple[int, int]:
        """Return (width, height) of the template image."""
        import cv2
        entry = self._registry.get_entry(template_id)
        img   = cv2.imread(entry["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            return (2480, 3508)   # A4 @ 300 DPI fallback
        h, w = img.shape[:2]
        return (w, h)


# ── Legacy compatibility ───────────────────────────────────────────────────────

def process_document(
    image_path:    str,
    template_size: tuple[int, int] = (2480, 3508),
) -> tuple[dict, dict]:
    """
    Original preprocessing-only entry point. Kept for backward compatibility.
    """
    image, h, w, aspect_ratio = load_image(image_path)
    gray = to_grayscale(image)

    stats: dict = {
        "original_width":  w,
        "original_height": h,
        "aspect_ratio":    aspect_ratio,
    }

    stats.update(baseline_metrics(gray))
    stats.update(skew_analysis(gray))

    normalized, illum = illumination_normalization(gray, stats["grayscale_std"])
    stats.update(illum)

    binary, bin_stats = binarization(normalized)
    stats.update(bin_stats)

    cropped, crop_stats = border_removal(binary, stats["threshold_stability"])
    stats.update(crop_stats)

    stats.update(structure_prep(cropped, template_size))
    stats["fusion_score"] = fuse(stats)

    return stats, {
        "gray":           gray,
        "normalized":     normalized,
        "binary":         binary,
        "cropped_binary": cropped,
    }