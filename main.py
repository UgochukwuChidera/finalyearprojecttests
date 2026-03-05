"""
DAPE Pipeline – entry point
============================

Step 1 – Auto-register a blank template (run once per template):
    python main.py --register

Step 2 – Process a completed form (or a whole batch):
    python main.py

Step 3 – Batch process:
    Uncomment the batch section at the bottom.
"""

import argparse
from project.template_analyzer.registry_builder import TemplateRegistryBuilder
from project.orchestrator import DAPEOrchestrator


# ── CLI args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="DAPE Pipeline")
parser.add_argument(
    "--register",
    action  = "store_true",
    help    = "Auto-register blank template(s) before processing",
)
args = parser.parse_args()


# ── Step 1 (optional) · Auto-register blank templates ─────────────────────────
if args.register:
    builder = TemplateRegistryBuilder(
        checkbox_max_area = 3000,
        textbox_min_area  = 3000,
        line_min_width    = 60,
        padding           = 4,
    )

    # Register a single template
    builder.build(
        template_id   = "template_001",
        image_path    = "templates/template_001.png",   # your blank form image
        registry_path = "templates/registry.json",
    )

    # ── Or register multiple templates at once ─────────────────────────────────
    # builder.build_batch(
    #     templates = [
    #         {"template_id": "template_001", "image_path": "templates/template_001.png"},
    #         {"template_id": "template_002", "image_path": "templates/template_002.png"},
    #         {"template_id": "template_003", "image_path": "templates/template_003.png"},
    #     ],
    #     registry_path = "templates/registry.json",
    # )

    print("\nAuto-registration complete. Review templates/registry.json if needed.")
    print("Now run:  python main.py  to process forms.\n")
    exit(0)


# ── Step 2 · Full DAPE pipeline ────────────────────────────────────────────────
orchestrator = DAPEOrchestrator(
    registry_path        = "templates/registry.json",
    output_dir           = "outputs",
    log_dir              = "logs",
    confidence_threshold = 0.60,
    enable_hitl          = True,
    hitl_host            = "127.0.0.1",
    hitl_port            = 5050,
)

result = orchestrator.process(
    image_path  = "form/678324.tif",
    template_id = "template_001",
)

print("=== DAPE Pipeline Result ===")
print(f"Form data  : {result['structured_output']['data']}")
print(f"Exports    : {result['export_paths']}")
print(f"Audit log  : {result['audit_log_path']}")
print(
    f"Escalation : flagged={result['stats'].get('esc_flagged_count', 0)} "
    f"corrected={result['stats'].get('esc_corrected_count', 0)}"
)


# ── Step 3 (optional) · Batch processing ──────────────────────────────────────
# import glob
# forms   = glob.glob("form/*.tif")
# results = orchestrator.process_batch(forms, template_id="template_001")
# for r in results:
#     if "error" in r:
#         print(f"ERROR {r['form_id']}: {r['error']}")
#     else:
#         print(f"OK    {r['structured_output']['form_id']}: "
#               f"exports → {r['export_paths']}")