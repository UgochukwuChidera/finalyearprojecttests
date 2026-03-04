import re

# ── Built-in format patterns ───────────────────────────────────────────────────
_PATTERNS: dict[str, re.Pattern] = {
    "date":         re.compile(r"^\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}$"),
    "email":        re.compile(r"^[\w\.\+\-]+@[\w\-]+\.[\w\.]{2,}$"),
    "phone":        re.compile(r"^\+?[\d\s\-\(\)]{7,15}$"),
    "numeric":      re.compile(r"^-?\d+(\.\d+)?$"),
    "integer":      re.compile(r"^-?\d+$"),
    "alpha":        re.compile(r"^[A-Za-z\s\-'\.]+$"),
    "alphanumeric": re.compile(r"^[A-Za-z0-9\s\-_\.]+$"),
    "postcode_uk":  re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$", re.I),
}


class SemanticValidator:
    """
    Applies field-level semantic validation rules to extracted string values.

    Validation checks include:
      - Required field presence
      - Format pattern matching (date, email, phone, etc.)
      - Maximum length constraint
      - Allowed-values whitelist

    Returns a validity flag and a human-readable reason code.
    """

    def validate(self, value: str, field_def: dict) -> dict:
        """
        Parameters
        ----------
        value     : str   – extracted (or corrected) field value as string
        field_def : dict  – template field definition; recognised keys:
            required      : bool         (default False)
            format        : str          one of the keys in _PATTERNS
            max_length    : int
            allowed_values: list[str]

        Returns
        -------
        dict with keys:
          valid  : bool
          reason : str  – short code explaining the outcome
        """
        required       = bool(field_def.get("required", False))
        format_type    = field_def.get("format")
        max_length     = field_def.get("max_length")
        allowed_values = field_def.get("allowed_values")

        stripped = value.strip() if isinstance(value, str) else ""

        if not stripped:
            if required:
                return {"valid": False, "reason": "required_field_empty"}
            return {"valid": True, "reason": "optional_empty"}

        if format_type and format_type in _PATTERNS:
            if not _PATTERNS[format_type].match(stripped):
                return {"valid": False, "reason": f"format_mismatch:{format_type}"}

        if max_length and len(stripped) > max_length:
            return {"valid": False, "reason": f"exceeds_max_length:{max_length}"}

        if allowed_values:
            lower_allowed = [v.lower() for v in allowed_values]
            if stripped.lower() not in lower_allowed:
                return {"valid": False, "reason": "value_not_in_allowed_set"}

        return {"valid": True, "reason": "passed"}
