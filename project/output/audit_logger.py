import json
from datetime import datetime, timezone
from pathlib import Path


class AuditLogger:
    """
    Records a structured audit entry for every processed form.

    Each log file captures:
      - Form and template identifiers
      - UTC timestamp
      - Pipeline processing statistics
      - Per-field summary (confidence, status, correction flag)
      - Escalation summary (total / flagged / corrected counts)

    Log entries are written as individual JSON files under *log_dir*,
    named by form_id + timestamp to avoid collisions.
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        form_id: str,
        template_id: str,
        processing_stats: dict,
        validated_fields: list[dict],
        export_paths: dict | None = None,
    ) -> str:
        """
        Write an audit entry and return the path to the log file.

        Parameters
        ----------
        form_id            : str
        template_id        : str
        processing_stats   : dict  – pipeline metrics from preprocessing / alignment
        validated_fields   : list  – final validated (and possibly corrected) fields
        export_paths       : dict  – optional {format: filepath} from DataExporter
        """
        ts       = datetime.now(timezone.utc)
        ts_label = ts.strftime("%Y%m%d_%H%M%S")

        field_summary = [
            {
                "field_id":          f["field_id"],
                "field_type":        f["field_type"],
                "confidence":        round(float(f.get("confidence", 0.0)), 4),
                "validation_status": f.get("validation_status", "unknown"),
                "validation_reason": f.get("validation_reason", ""),
                "corrected":         f.get("corrected", False),
            }
            for f in validated_fields
        ]

        total     = len(validated_fields)
        flagged   = sum(1 for f in validated_fields if f.get("needs_review",  False))
        corrected = sum(1 for f in validated_fields if f.get("corrected",     False))
        accepted  = sum(1 for f in validated_fields if f.get("validation_status") == "accepted")

        entry = {
            "form_id":       form_id,
            "template_id":   template_id,
            "timestamp":     ts.isoformat(),
            "processing_stats": {
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in processing_stats.items()
                if not isinstance(v, (bytes, type(None)))
            },
            "field_summary": field_summary,
            "escalation_summary": {
                "total_fields":    total,
                "accepted":        accepted,
                "flagged":         flagged,
                "corrected":       corrected,
                "escalation_rate": round(flagged / total, 4) if total else 0.0,
                "auto_accept_rate": round(accepted / total, 4) if total else 0.0,
            },
            "export_paths": export_paths or {},
        }

        log_path = self.log_dir / f"{form_id}_{ts_label}.json"
        with log_path.open("w", encoding="utf-8") as fh:
            json.dump(entry, fh, indent=2, default=str, ensure_ascii=False)

        return str(log_path)
