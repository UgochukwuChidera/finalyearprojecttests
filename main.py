from project.orchestrator import process_document

file_path = "form/678324.tif"

stats, processed_images = process_document(file_path)

for k, v in stats.items():
    print(f"{k} -> {v}")