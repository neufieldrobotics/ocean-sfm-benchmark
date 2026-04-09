"""DISK feature matcher with brute-force descriptor matching."""

import cv2
import numpy as np
import torch
from matchers.base import BaseMatcher
from config import MAX_IMAGE_DIM, DISK_MAX_KEYPOINTS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DISKBFMatcher(BaseMatcher):
    """DISK features with brute-force KNN matching + ratio test."""
    name = "DISK"

    def __init__(self, max_keypoints=DISK_MAX_KEYPOINTS, ratio_thresh=0.8):
        self.max_keypoints = max_keypoints
        self.ratio_thresh = ratio_thresh
        self.extractor = None
        self._init_matcher()

    def _init_matcher(self):
        try:
            from lightglue import DISK
            kwargs = {}
            if self.max_keypoints > 0:
                kwargs["max_num_keypoints"] = self.max_keypoints
            self.extractor = DISK(**kwargs).eval().to(device)
        except Exception as e:
            print(f"DISK init failed: {e}")

    @staticmethod
    def _read_image_tensor(path):
        import torchvision.transforms.functional as TF
        from PIL import Image

        img = Image.open(str(path)).convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        tensor = TF.to_tensor(img).unsqueeze(0).to(device)
        return tensor

    def match(self, path0, path1):
        if self.extractor is None:
            return np.array([]), np.array([])

        tensor0 = self._read_image_tensor(path0)
        tensor1 = self._read_image_tensor(path1)

        with torch.no_grad():
            feats0 = self.extractor.extract(tensor0)
            feats1 = self.extractor.extract(tensor1)

        kpts0 = feats0["keypoints"][0].cpu().numpy()
        kpts1 = feats1["keypoints"][0].cpu().numpy()
        desc0 = feats0["descriptors"][0].cpu().numpy()
        desc1 = feats1["descriptors"][0].cpu().numpy()

        if len(kpts0) == 0 or len(kpts1) == 0:
            return np.array([]), np.array([])

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(desc0.astype(np.float32),
                              desc1.astype(np.float32), k=2)

        good = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_thresh * n.distance:
                    good.append(m)

        if not good:
            return np.array([]), np.array([])

        mkpts0 = np.array([kpts0[m.queryIdx] for m in good])
        mkpts1 = np.array([kpts1[m.trainIdx] for m in good])
        return mkpts0, mkpts1
