"""RoMa dense matcher for benchmark."""

import cv2
import numpy as np
import torch
from matchers.base import BaseMatcher

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RoMaMatcher(BaseMatcher):
    name = "RoMa"

    def __init__(self, num_samples=5000):
        self.num_samples = num_samples
        self.roma = None
        self._init_matcher()

    def _init_matcher(self):
        try:
            from romatch import roma_outdoor
            self.roma = roma_outdoor(device=device)
        except Exception as e:
            print(f"RoMa init failed: {e}")

    def match(self, path0, path1):
        if self.roma is None:
            return np.array([]), np.array([])

        img0 = cv2.imread(str(path0))
        img1 = cv2.imread(str(path1))
        H0, W0 = img0.shape[:2]
        H1, W1 = img1.shape[:2]

        with torch.no_grad():
            warp, certainty = self.roma.match(path0, path1, device=device)
            matches, _ = self.roma.sample(warp, certainty, num=self.num_samples)

            mkpts0 = matches[:, :2].cpu().numpy()
            mkpts1 = matches[:, 2:].cpu().numpy()

            mkpts0[:, 0] = (mkpts0[:, 0] + 1) / 2 * W0
            mkpts0[:, 1] = (mkpts0[:, 1] + 1) / 2 * H0
            mkpts1[:, 0] = (mkpts1[:, 0] + 1) / 2 * W1
            mkpts1[:, 1] = (mkpts1[:, 1] + 1) / 2 * H1

            valid = (
                (mkpts0[:, 0] >= 0) & (mkpts0[:, 0] < W0) &
                (mkpts0[:, 1] >= 0) & (mkpts0[:, 1] < H0) &
                (mkpts1[:, 0] >= 0) & (mkpts1[:, 0] < W1) &
                (mkpts1[:, 1] >= 0) & (mkpts1[:, 1] < H1)
            )
            mkpts0 = mkpts0[valid]
            mkpts1 = mkpts1[valid]

        return mkpts0, mkpts1
