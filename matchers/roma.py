"""RoMa dense matcher for benchmark."""

import cv2
import numpy as np
import torch
from PIL import Image
from matchers.base import BaseMatcher
from config import MAX_IMAGE_DIM, MAX_MATCHES_PER_PAIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RoMaMatcher(BaseMatcher):
    name = "RoMa"

    def __init__(self, num_samples=MAX_MATCHES_PER_PAIR, variant="tiny"):
        self.num_samples = num_samples
        self.variant = variant
        self.roma = None
        if variant == "full":
            self.name = "RoMa-full"
        self._init_matcher()

    def _init_matcher(self):
        try:
            if self.variant == "full":
                from romatch import roma_outdoor
                self.roma = roma_outdoor(device=device)
            else:
                from romatch import tiny_roma_v1_outdoor
                self.roma = tiny_roma_v1_outdoor(device=device)
                print(f"Initialized RoMa ({self.variant}) on {device}")
        except Exception as e:
            print(f"RoMa ({self.variant}) init failed: {e}")

    @staticmethod
    def _load_resized_pil(path):
        """Load image as PIL, resize to MAX_IMAGE_DIM, round dims to multiples of 14."""
        img = Image.open(str(path)).convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(w, h)
            w, h = int(w * scale), int(h * scale)
        w = (w // 14) * 14
        h = (h // 14) * 14
        return img.resize((w, h), Image.LANCZOS)

    def match(self, path0, path1):
        if self.roma is None:
            return np.array([]), np.array([])

        img0 = cv2.imread(str(path0))
        img1 = cv2.imread(str(path1))
        H0, W0 = img0.shape[:2]
        H1, W1 = img1.shape[:2]

        with torch.no_grad():
            im0 = self._load_resized_pil(path0)
            im1 = self._load_resized_pil(path1)
            warp, certainty = self.roma.match(im0, im1, batched=True)
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
