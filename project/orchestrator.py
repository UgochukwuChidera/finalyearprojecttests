from .preprocessing.io import load_image
from .preprocessing.grayscale import to_grayscale
from .preprocessing.baseline_metrics import baseline_metrics
from .preprocessing.skew_analysis import skew_analysis
from .preprocessing.illumination import illumination_normalization
from .preprocessing.binarization import binarization
from .preprocessing.border_removal import border_removal
from .preprocessing.structure_prep import structure_prep
from .preprocessing.fusion import fuse


def process_document(image_path, template_size=(2480, 3508)):
    image, h, w, aspect_ratio = load_image(image_path)
    gray = to_grayscale(image)

    stats = {
        "original_width": w,
        "original_height": h,
        "aspect_ratio": aspect_ratio
    }

    stats.update(baseline_metrics(gray))
    stats.update(skew_analysis(gray))

    normalized, illum = illumination_normalization(
        gray, stats["grayscale_std"]
    )
    stats.update(illum)

    binary, bin_stats = binarization(normalized)
    stats.update(bin_stats)

    cropped, crop_stats = border_removal(
        binary, stats["threshold_stability"]
    )
    stats.update(crop_stats)

    stats.update(structure_prep(cropped, template_size))
    stats["fusion_score"] = fuse(stats)

    return stats, {
        "gray": gray,
        "normalized": normalized,
        "binary": binary,
        "cropped_binary": cropped
    }