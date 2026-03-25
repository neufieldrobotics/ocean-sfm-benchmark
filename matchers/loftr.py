"""LoFTR dense matcher for benchmark."""

import numpy as np
import torch
from matchers.base import BaseMatcher
from matchers.sift import load_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LoFTRMatcher(BaseMatcher):
    name = "LoFTR"

    def __init__(self, confidence_thresh=0.5):
        self.confidence_thresh = confidence_thresh
        self.loftr = None
        self._init_matcher()

    def _init_matcher(self):
        try:
            from kornia.feature import LoFTR
            self.loftr = LoFTR(pretrained="outdoor").eval().to(device)
        except Exception as e:
            print(f"LoFTR init failed: {e}")

    def match(self, path0, path1):
        if self.loftr is None:
            return np.array([]), np.array([])

        _, gray0, _ = load_image(path0, resize=840)
        _, gray1, _ = load_image(path1, resize=840)

        # LoFTR requires dimensions divisible by 8
        h0, w0 = (gray0.shape[0] // 8) * 8, (gray0.shape[1] // 8) * 8
        h1, w1 = (gray1.shape[0] // 8) * 8, (gray1.shape[1] // 8) * 8
        gray0, gray1 = gray0[:h0, :w0], gray1[:h1, :w1]

        tensor0 = torch.from_numpy(gray0).float()[None, None].to(device) / 255.0
        tensor1 = torch.from_numpy(gray1).float()[None, None].to(device) / 255.0

        with torch.no_grad():
            result = self.loftr({"image0": tensor0, "image1": tensor1})

        mkpts0 = result["keypoints0"].cpu().numpy()
        mkpts1 = result["keypoints1"].cpu().numpy()
        conf = result["confidence"].cpu().numpy()

        mask = conf > self.confidence_thresh
        return mkpts0[mask], mkpts1[mask]
