class HITLEscalation:
    """
    Manages selective human-in-the-loop correction for flagged form fields.

    Human corrections update *final_value* and set *corrected = True* on
    the affected fields. They do NOT retrain models, alter confidence
    thresholds, or modify any processing logic within the pipeline.
    """

    # ── Inspection helpers ─────────────────────────────────────────────────────

    @staticmethod
    def get_flagged(validated_fields: list[dict]) -> list[dict]:
        """Return only fields that require human review."""
        return [f for f in validated_fields if f.get("needs_review", False)]

    # ── Correction application ─────────────────────────────────────────────────

    @staticmethod
    def apply_correction(
        field_id: str,
        corrected_value,
        validated_fields: list[dict],
    ) -> list[dict]:
        """
        Apply a single human correction to *validated_fields*.

        Returns a new list; the original is not mutated.
        """
        updated = []
        for field in validated_fields:
            if field["field_id"] == field_id:
                field = {
                    **field,
                    "final_value":       corrected_value,
                    "corrected":         True,
                    "validation_status": "corrected",
                    "needs_review":      False,
                }
            updated.append(field)
        return updated

    @classmethod
    def apply_bulk_corrections(
        cls,
        corrections: dict,
        validated_fields: list[dict],
    ) -> list[dict]:
        """
        Apply a mapping of {field_id: corrected_value} in one pass.

        Parameters
        ----------
        corrections : dict
            Keys are field IDs; values are the human-provided corrections.
        validated_fields : list[dict]
            Current validated field list.

        Returns
        -------
        Updated validated field list.
        """
        for fid, value in corrections.items():
            validated_fields = cls.apply_correction(fid, value, validated_fields)
        return validated_fields

    # ── Summary statistics ─────────────────────────────────────────────────────

    @staticmethod
    def escalation_stats(validated_fields: list[dict]) -> dict:
        """Return summary counts for audit logging."""
        total     = len(validated_fields)
        flagged   = sum(1 for f in validated_fields if f.get("needs_review",  False))
        corrected = sum(1 for f in validated_fields if f.get("corrected",     False))
        return {
            "total_fields":     total,
            "flagged_count":    flagged,
            "corrected_count":  corrected,
            "escalation_rate":  round(flagged / total, 4) if total else 0.0,
        }
