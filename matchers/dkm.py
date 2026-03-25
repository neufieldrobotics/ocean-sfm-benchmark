"""DKM dense matcher for benchmark."""

import cv2
import numpy as np
import torch
from matchers.base import BaseMatcher

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DKMMatcher(BaseMatcher):
    name = "DKM"

    def __init__(self, num_samples=5000):
        self.num_samples = num_samples
        self.dkm = None
        self._init_matcher()

    def _init_matcher(self):
        try:
            # Try various DKM API versions
            try:
                from dkm import DKMv3_outdoor
                self.dkm = DKMv3_outdoor(device=device)
                return
            except (ImportError, AttributeError):
                pass
            try:
                from dkm import DKMv3
                self.dkm = DKMv3(device=device, outdoor=True)
                return
            except (ImportError, AttributeError):
                pass
            try:
                from dkm import dkm_base
                self.dkm = dkm_base(device=device)
                return
            except (ImportError, AttributeError):
                pass
            print("DKM init failed: no compatible DKM API found. "
                  "Install via: pip install dkm (the CV matching package)")
        except Exception as e:
            print(f"DKM init failed: {e}")

    def match(self, path0, path1):
        if self.dkm is None:
            return np.array([]), np.array([])

        img0 = cv2.imread(str(path0))
        img1 = cv2.imread(str(path1))
        H0, W0 = img0.shape[:2]
        H1, W1 = img1.shape[:2]

        with torch.no_grad():
            dense_matches, dense_certainty = self.dkm.match(
                path0, path1, device=device)
            sparse_matches, _ = self.dkm.sample(
                dense_matches, dense_certainty, num=self.num_samples)

        mkpts0 = sparse_matches[:, :2].cpu().numpy()
        mkpts1 = sparse_matches[:, 2:].cpu().numpy()

        mkpts0[:, 0] = (mkpts0[:, 0] + 1) / 2 * W0
        mkpts0[:, 1] = (mkpts0[:, 1] + 1) / 2 * H0
        mkpts1[:, 0] = (mkpts1[:, 0] + 1) / 2 * W1
        mkpts1[:, 1] = (mkpts1[:, 1] + 1) / 2 * H1

        return mkpts0, mkpts1
