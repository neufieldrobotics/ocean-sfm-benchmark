"""DKM dense matcher for benchmark."""

import os
import tempfile
import cv2
import numpy as np
import torch
from PIL import Image
from matchers.base import BaseMatcher
from config import MAX_IMAGE_DIM, MAX_MATCHES_PER_PAIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DKMMatcher(BaseMatcher):
    name = "DKM"

    def __init__(self, num_samples=MAX_MATCHES_PER_PAIR):
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

    @staticmethod
    def _resize_to_temp(path):
        """Load image, resize to MAX_IMAGE_DIM, save to temp file for DKM's path-based API."""
        img = Image.open(str(path)).convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        img.save(tmp_path)
        return tmp_path

    def match(self, path0, path1):
        if self.dkm is None:
            return np.array([]), np.array([])

        img0 = cv2.imread(str(path0))
        img1 = cv2.imread(str(path1))
        H0, W0 = img0.shape[:2]
        H1, W1 = img1.shape[:2]

        # Resize to MAX_IMAGE_DIM via temp files (DKM API is path-based)
        tmp0 = self._resize_to_temp(path0)
        tmp1 = self._resize_to_temp(path1)
        try:
            with torch.no_grad():
                dense_matches, dense_certainty = self.dkm.match(
                    tmp0, tmp1, device=device)
                sparse_matches, _ = self.dkm.sample(
                    dense_matches, dense_certainty, num=self.num_samples)
        finally:
            os.unlink(tmp0)
            os.unlink(tmp1)

        mkpts0 = sparse_matches[:, :2].cpu().numpy()
        mkpts1 = sparse_matches[:, 2:].cpu().numpy()

        # Map normalized [-1,1] coords to original image space
        mkpts0[:, 0] = (mkpts0[:, 0] + 1) / 2 * W0
        mkpts0[:, 1] = (mkpts0[:, 1] + 1) / 2 * H0
        mkpts1[:, 0] = (mkpts1[:, 0] + 1) / 2 * W1
        mkpts1[:, 1] = (mkpts1[:, 1] + 1) / 2 * H1

        return mkpts0, mkpts1
