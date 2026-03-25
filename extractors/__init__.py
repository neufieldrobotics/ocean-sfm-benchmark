"""Feature extractor registry for COLMAP pipeline."""

from extractors.superpoint_superglue import SuperPointSuperGlueExtractor
from extractors.lightglue_extractor import LightGlueExtractor
from extractors.dense_extractor import DenseExtractor
from extractors.sift_native import SIFTNativeExtractor


def get_extractor(method, device="cuda"):
    """Create an extractor instance by method name.

    Args:
        method: One of the supported method names.
        device: torch device string.

    Returns:
        Extractor instance, or SIFTNativeExtractor for "sift".
    """
    factories = {
        "superpoint+superglue": lambda: SuperPointSuperGlueExtractor(device),
        "superpoint+lightglue": lambda: LightGlueExtractor("superpoint", device),
        "aliked+lightglue": lambda: LightGlueExtractor("aliked", device),
        "disk+lightglue": lambda: LightGlueExtractor("disk", device),
        "loftr": lambda: DenseExtractor("loftr", device),
        "roma": lambda: DenseExtractor("roma", device),
        "dkm": lambda: DenseExtractor("dkm", device),
        "sift": lambda: SIFTNativeExtractor(),
    }

    if method not in factories:
        raise ValueError(
            f"Unknown method: {method}. Available: {list(factories.keys())}")

    return factories[method]()


AVAILABLE_METHODS = [
    "sift",
    "superpoint+superglue",
    "superpoint+lightglue",
    "aliked+lightglue",
    "disk+lightglue",
    "loftr",
    "roma",
    "dkm",
]
