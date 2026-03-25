"""Base class for feature extractors used in COLMAP pipeline."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple
import numpy as np


class BaseExtractor(ABC):
    """Interface for feature extraction + matching for COLMAP integration.

    Sparse extractors (SuperPoint, ALIKED, DISK) produce per-image keypoints
    and descriptors. Dense extractors (LoFTR, RoMa, DKM) produce per-pair
    pixel correspondences instead.
    """
    name: str = "base"
    is_dense: bool = False

    def __init__(self, device="cuda"):
        self.device = device

    @abstractmethod
    def extract_features_image(self, image_path: Path) -> dict:
        """Extract features for one image.

        For sparse extractors, returns:
            {"kps_orig": np.ndarray (N,2), "desc": np.ndarray (N,D),
             "scores": np.ndarray (N,), "orig_hw": (H, W), ...}
        For dense extractors, returns:
            {"orig_hw": (H, W), "image_path": Path}
        """

    @abstractmethod
    def match_pair(self, feat0: dict, feat1: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Match a cached feature pair.

        For sparse extractors:
            Returns (matches [M, 2] index pairs, confidence [M])
        For dense extractors:
            Returns (pts0 [M, 2] pixel coords, pts1 [M, 2] pixel coords,
                     confidence [M])
        """
