from .semantic_validator import SemanticValidator

# Validation status codes
_ACCEPTED        = "accepted"
_LOW_CONFIDENCE  = "low_confidence"
_SEMANTIC_FAIL   = "semantic_failure"
_REJECTED        = "rejected"


class ConfidenceValidator:
    """
    Assigns a validation status to each extracted field by combining:
      1. Confidence score threshold check
      2. Semantic format / constraint validation

    Fields that pass both checks are accepted automatically.
    All others are flagged for human-in-the-loop review.

    Validation does not modify extraction logic or system parameters.
    """

    def __init__(self, confidence_threshold: float = 0.60):
        self.threshold = confidence_threshold
        self._semantic = SemanticValidator()

    def validate(
        self,
        extracted_fields: list[dict],
        field_definitions: list[dict],
    ) -> list[dict]:
        """
        Validate every extracted field and attach status metadata.

        Parameters
        ----------
        extracted_fields   : output of FieldExtractor.extract_fields()
        field_definitions  : list of template field defs (same set)

        Returns
        -------
        list[dict] – each input dict extended with:
          validation_status  : str   – see status codes above
          validation_reason  : str   – from SemanticValidator or threshold
          needs_review       : bool  – True → route to HITL
          final_value        : any   – initially equal to extracted value
          corrected          : bool  – False until HITL correction applied
        """
        def_map = {f["id"]: f for f in field_definitions}
        results = []

        for field in extracted_fields:
            fid        = field["field_id"]
            confidence = float(field.get("confidence", 0.0))
            value      = field.get("value", "")
            fdef       = def_map.get(fid, {})

            # Semantic check (only meaningful for text fields)
            if field.get("field_type") != "checkbox":
                sem = self._semantic.validate(str(value), fdef)
            else:
                sem = {"valid": True, "reason": "checkbox_field"}

            above_threshold  = confidence >= self.threshold
            semantically_ok  = sem["valid"]

            if above_threshold and semantically_ok:
                status       = _ACCEPTED
                needs_review = False
            elif above_threshold and not semantically_ok:
                status       = _SEMANTIC_FAIL
                needs_review = True
            elif not above_threshold and semantically_ok:
                status       = _LOW_CONFIDENCE
                needs_review = True
            else:
                status       = _REJECTED
                needs_review = True

            results.append({
                **field,
                "validation_status":  status,
                "validation_reason":  sem["reason"],
                "needs_review":       needs_review,
                "final_value":        value,
                "corrected":          False,
            })

        return results
