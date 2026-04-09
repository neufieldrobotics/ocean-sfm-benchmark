"""Unified LightGlue extractor for SuperPoint/ALIKED/DISK + LightGlue.

Supports three feature types via parameterization:
    LightGlueExtractor("superpoint")
    LightGlueExtractor("aliked")
    LightGlueExtractor("disk")
"""

import numpy as np
import torch
from pathlib import Path

from extractors.base import BaseExtractor
from config import (
    MAX_IMAGE_DIM, SP_MAX_KEYPOINTS, ALIKED_MAX_KEYPOINTS,
    ALIKED_DETECTION_THRESHOLD, ALIKED_NMS_RADIUS, DISK_MAX_KEYPOINTS,
)


class LightGlueExtractor(BaseExtractor):
    """Unified extractor for SuperPoint/ALIKED/DISK features + LightGlue matcher."""

    def __init__(self, feature_type="aliked", device="cuda"):
        super().__init__(device)
        assert feature_type in ("superpoint", "aliked", "disk"), \
            f"Unsupported feature_type: {feature_type}"
        self.feature_type = feature_type
        self.name = f"{feature_type}+lightglue"
        self._setup_models()

    def _setup_models(self):
        from lightglue import LightGlue

        if self.feature_type == "superpoint":
            from lightglue import SuperPoint
            self.extractor = SuperPoint(
                max_num_keypoints=SP_MAX_KEYPOINTS if SP_MAX_KEYPOINTS > 0 else -1,
            ).eval().to(self.device)
        elif self.feature_type == "aliked":
            from lightglue import ALIKED
            self.extractor = ALIKED(
                max_num_keypoints=ALIKED_MAX_KEYPOINTS if ALIKED_MAX_KEYPOINTS > 0 else -1,
                detection_threshold=ALIKED_DETECTION_THRESHOLD,
                nms_radius=ALIKED_NMS_RADIUS,
            ).eval().to(self.device)
        elif self.feature_type == "disk":
            from lightglue import DISK
            self.extractor = DISK(
                max_num_keypoints=DISK_MAX_KEYPOINTS if DISK_MAX_KEYPOINTS > 0 else -1,
            ).eval().to(self.device)

        self.matcher = LightGlue(
            features=self.feature_type,
            depth_confidence=0.95,
            width_confidence=0.99,
        ).eval().to(self.device)

        print(f"{self.feature_type.upper()} + LightGlue loaded on {self.device}")

    @staticmethod
    def _read_image_tensor(path, device):
        """Read image as [1, 3, H, W] tensor in [0, 1] range, resized to MAX_IMAGE_DIM."""
        import torchvision.transforms.functional as TF
        from PIL import Image

        img = Image.open(str(path)).convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        tensor = TF.to_tensor(img).unsqueeze(0).to(device)
        return tensor, img.size  # (W, H) after resize

    def extract_features_image(self, image_path):
        img_tensor, (w_orig, h_orig) = self._read_image_tensor(image_path, self.device)

        with torch.no_grad():
            feats = self.extractor.extract(img_tensor)

        kps = feats["keypoints"][0].detach().cpu().numpy().astype(np.float32)
        desc = feats["descriptors"][0].detach().cpu().numpy().astype(np.float32)
        scores = feats["keypoint_scores"][0].detach().cpu().numpy().astype(np.float32)

        # Map keypoints back to original image coordinates
        proc_size = feats["image_size"][0].detach().cpu().numpy()
        w_proc, h_proc = float(proc_size[0]), float(proc_size[1])
        scale_x = w_orig / w_proc
        scale_y = h_orig / h_proc

        kps_orig = kps.copy()
        if abs(scale_x - 1.0) > 1e-3 or abs(scale_y - 1.0) > 1e-3:
            kps_orig[:, 0] *= scale_x
            kps_orig[:, 1] *= scale_y

        return {
            "feats": feats,
            "kps_orig": kps_orig,
            "desc": desc,
            "scores": scores,
            "orig_hw": (h_orig, w_orig),
        }

    def match_pair(self, feat0, feat1):
        with torch.no_grad():
            result = self.matcher({"image0": feat0["feats"], "image1": feat1["feats"]})

        matches = result["matches"][0].detach().cpu().numpy()
        scores = result.get("matching_scores0", [None])[0]
        if scores is None:
            scores = result.get("scores", [None])[0]

        if matches is None or len(matches) == 0:
            return None, None

        matches = matches.astype(np.int32)

        if scores is not None and hasattr(scores, "detach"):
            scores = scores.detach().cpu().numpy().astype(np.float32)
        elif scores is None:
            scores = np.ones(len(matches), dtype=np.float32)

        return matches, scores
