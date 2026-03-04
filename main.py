"""
DAPE Pipeline – entry point examples
=====================================

Full pipeline (all 8 stages):
    python main.py

Preprocessing-only (original behaviour, unchanged):
    from project.orchestrator import process_document
    stats, images = process_document("form/678324.tif")
"""

from project.orchestrator import DAPEOrchestrator

# ── Full DAPE pipeline ─────────────────────────────────────────────────────────

orchestrator = DAPEOrchestrator(
    registry_path       = "templates/registry.json",
    output_dir          = "outputs",
    log_dir             = "logs",
    confidence_threshold = 0.60,
    enable_hitl         = True,       # set False to skip human review
    hitl_host           = "127.0.0.1",
    hitl_port           = 5050,
)

result = orchestrator.process(
    image_path  = "form/678324.tif",
    template_id = "template_001",     # must exist in templates/registry.json
)

print("=== DAPE Pipeline Result ===")
print(f"Form data  : {result['structured_output']['data']}")
print(f"Exports    : {result['export_paths']}")
print(f"Audit log  : {result['audit_log_path']}")
print(f"Escalation : flagged={result['stats'].get('esc_flagged_count', 0)} "
      f"corrected={result['stats'].get('esc_corrected_count', 0)}")


# ── Batch processing example ──────────────────────────────────────────────────

# import glob
# forms = glob.glob("form/*.tif")
# results = orchestrator.process_batch(forms, template_id="template_001")
# for r in results:
#     if "error" in r:
#         print(f"ERROR {r['form_id']}: {r['error']}")
#     else:
#         print(f"OK    {r['structured_output']['form_id']}: "
#               f"exports → {r['export_paths']}")
