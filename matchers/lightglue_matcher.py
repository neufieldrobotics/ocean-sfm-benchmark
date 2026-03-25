"""Unified LightGlue benchmark matcher for SuperPoint/ALIKED/DISK + LightGlue."""

import numpy as np
import torch
from matchers.base import BaseMatcher

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LightGlueMatcher(BaseMatcher):
    """Benchmark matcher for any LightGlue-supported feature type.

    Usage:
        LightGlueMatcher("superpoint")  -> name: "SP+LG"
        LightGlueMatcher("aliked")      -> name: "ALIKED+LG"
        LightGlueMatcher("disk")        -> name: "DISK+LG"
    """

    _DISPLAY_NAMES = {
        "superpoint": "SP+LG",
        "aliked": "ALIKED+LG",
        "disk": "DISK+LG",
    }

    def __init__(self, feature_type="superpoint"):
        assert feature_type in ("superpoint", "aliked", "disk")
        self.feature_type = feature_type
        self.name = self._DISPLAY_NAMES[feature_type]
        self.extractor = None
        self.matcher = None
        self._init_matcher()

    def _init_matcher(self):
        try:
            from lightglue import LightGlue

            if self.feature_type == "superpoint":
                from lightglue import SuperPoint
                self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
            elif self.feature_type == "aliked":
                from lightglue import ALIKED
                self.extractor = ALIKED(max_num_keypoints=4096).eval().to(device)
            elif self.feature_type == "disk":
                from lightglue import DISK
                self.extractor = DISK(max_num_keypoints=4096).eval().to(device)

            self.matcher = LightGlue(features=self.feature_type).eval().to(device)
        except Exception as e:
            print(f"{self.name} init failed: {e}")

    @staticmethod
    def _read_image_tensor(path):
        import torchvision.transforms.functional as TF
        from PIL import Image

        img = Image.open(str(path)).convert("RGB")
        # Resize if too large for benchmark
        w, h = img.size
        max_dim = 1024
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        tensor = TF.to_tensor(img).unsqueeze(0).to(device)
        return tensor

    def match(self, path0, path1):
        if self.extractor is None or self.matcher is None:
            return np.array([]), np.array([])

        tensor0 = self._read_image_tensor(path0)
        tensor1 = self._read_image_tensor(path1)

        with torch.no_grad():
            feats0 = self.extractor.extract(tensor0)
            feats1 = self.extractor.extract(tensor1)
            result = self.matcher({"image0": feats0, "image1": feats1})

        matches = result["matches"][0].cpu().numpy()
        if matches is None or len(matches) == 0:
            return np.array([]), np.array([])

        kpts0 = feats0["keypoints"][0].cpu().numpy()
        kpts1 = feats1["keypoints"][0].cpu().numpy()

        mkpts0 = kpts0[matches[:, 0]]
        mkpts1 = kpts1[matches[:, 1]]

        return mkpts0, mkpts1
