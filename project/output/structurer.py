from datetime import datetime, timezone


class OutputStructurer:
    """
    Maps validated field values to predefined semantic keys and assembles
    a standardised JSON-compatible output document.

    JSON is the canonical output representation. Auxiliary formats (CSV,
    XLSX) are produced by DataExporter from this structured document.
    """

    def __init__(self, output_schema: dict[str, str]):
        """
        Parameters
        ----------
        output_schema : dict
            Maps field_id → semantic_key, e.g.
            {"q1_name": "respondent_name", "q2_date": "submission_date"}
        """
        self.schema = output_schema

    def structure(
        self,
        validated_fields: list[dict],
        form_id: str,
        template_id: str,
        processing_stats: dict | None = None,
    ) -> dict:
        """
        Build the canonical output document.

        Returns
        -------
        dict with keys:
          form_id          : str
          template_id      : str
          processed_at     : ISO-8601 UTC timestamp
          data             : dict – semantic_key → final_value
          fields           : list – full validated field records
          processing_stats : dict – pipeline metadata (optional)
        """
        data: dict = {}
        for field in validated_fields:
            fid          = field["field_id"]
            semantic_key = self.schema.get(fid, fid)
            data[semantic_key] = field.get("final_value", "")

        return {
            "form_id":          form_id,
            "template_id":      template_id,
            "processed_at":     datetime.now(timezone.utc).isoformat(),
            "data":             data,
            "fields":           validated_fields,
            "processing_stats": processing_stats or {},
        }
