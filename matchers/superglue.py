"""SuperPoint + SuperGlue benchmark matcher."""

import sys
import numpy as np
import torch
from typing import Tuple
from matchers.base import BaseMatcher
from matchers.sift import load_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SuperGlueMatcher(BaseMatcher):
    name = "SuperGlue"

    def __init__(self):
        self.matching = None
        self._init_matcher()

    def _init_matcher(self):
        try:
            sys.path.append("./SuperGluePretrainedNetwork")
            from models.matching import Matching
            config = {
                "superpoint": {
                    "nms_radius": 4,
                    "keypoint_threshold": 0.005,
                    "max_keypoints": 2048,
                },
                "superglue": {
                    "weights": "outdoor",
                    "sinkhorn_iterations": 20,
                    "match_threshold": 0.2,
                },
            }
            self.matching = Matching(config).eval().to(device)
        except Exception as e:
            print(f"SuperGlue init failed: {e}")

    def match(self, path0, path1):
        if self.matching is None:
            return np.array([]), np.array([])

        from models.utils import frame2tensor

        _, gray0, _ = load_image(path0)
        _, gray1, _ = load_image(path1)

        tensor0 = frame2tensor(gray0, device)
        tensor1 = frame2tensor(gray1, device)

        with torch.no_grad():
            pred = self.matching({"image0": tensor0, "image1": tensor1})

        kpts0 = pred["keypoints0"][0].cpu().numpy()
        kpts1 = pred["keypoints1"][0].cpu().numpy()
        matches = pred["matches0"][0].cpu().numpy()

        valid = matches > -1
        return kpts0[valid], kpts1[matches[valid]]
