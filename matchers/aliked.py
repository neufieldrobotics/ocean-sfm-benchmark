"""ALIKED feature matcher with brute-force descriptor matching."""

import sys
import cv2
import numpy as np
import torch
from matchers.base import BaseMatcher
from matchers.sift import load_image
from config import MAX_IMAGE_DIM, ALIKED_MAX_KEYPOINTS, ALIKED_DETECTION_THRESHOLD, ALIKED_NMS_RADIUS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ALIKEDMatcher(BaseMatcher):
    """ALIKED with BF descriptor matching (ratio test)."""
    name = "ALIKED"

    def __init__(self, max_keypoints=ALIKED_MAX_KEYPOINTS,
                 detection_threshold=ALIKED_DETECTION_THRESHOLD,
                 ratio_thresh=0.8, model_name="aliked-n16rot"):
        self.max_keypoints = max_keypoints
        self.detection_threshold = detection_threshold
        self.ratio_thresh = ratio_thresh
        self.model_name = model_name
        self.aliked = None
        self.backend = None
        self._init_matcher()

    def _init_matcher(self):
        # Try lightglue's ALIKED first (most reliable)
        try:
            from lightglue import ALIKED
            kwargs = {"detection_threshold": self.detection_threshold}
            if self.max_keypoints > 0:
                kwargs["max_num_keypoints"] = self.max_keypoints
            self.aliked = ALIKED(**kwargs).eval().to(device)
            self.backend = "lightglue"
            return
        except (ImportError, Exception):
            pass

        # Try kornia
        try:
            from kornia.feature import ALIKED as ALIKED_Kornia
            kornia_kwargs = {
                "detection_threshold": self.detection_threshold,
                "model_name": self.model_name,
            }
            if self.max_keypoints > 0:
                kornia_kwargs["max_num_keypoints"] = self.max_keypoints
            self.aliked = ALIKED_Kornia(**kornia_kwargs).eval().to(device)
            self.backend = "kornia"
            return
        except (ImportError, Exception):
            pass

        # Fallback to standalone repo
        try:
            sys.path.append("./ALIKED")
            from nets.aliked import ALIKED as ALIKED_Standalone
            standalone_kwargs = {
                "model_name": self.model_name,
                "scores_th": self.detection_threshold,
            }
            if self.max_keypoints > 0:
                standalone_kwargs["top_k"] = self.max_keypoints
                standalone_kwargs["n_limit"] = self.max_keypoints
            self.aliked = ALIKED_Standalone(**standalone_kwargs).eval().to(device)
            self.backend = "standalone"
        except Exception as e:
            print(f"ALIKED init failed: {e}")

    def match(self, path0, path1):
        if self.aliked is None:
            return np.array([]), np.array([])

        if self.backend == "lightglue":
            return self._match_lightglue(path0, path1)
        elif self.backend == "kornia":
            return self._match_kornia(path0, path1)
        return self._match_standalone(path0, path1)

    def _match_lightglue(self, path0, path1):
        """Match using lightglue's ALIKED extractor."""
        import torchvision.transforms.functional as TF
        from PIL import Image

        def _load(path):
            img = Image.open(str(path)).convert("RGB")
            w, h = img.size
            if max(w, h) > MAX_IMAGE_DIM:
                scale = MAX_IMAGE_DIM / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            return TF.to_tensor(img).unsqueeze(0).to(device)

        tensor0 = _load(path0)
        tensor1 = _load(path1)

        with torch.no_grad():
            feats0 = self.aliked.extract(tensor0)
            feats1 = self.aliked.extract(tensor1)

        kpts0 = feats0["keypoints"][0].cpu().numpy()
        kpts1 = feats1["keypoints"][0].cpu().numpy()
        desc0 = feats0["descriptors"][0].cpu().numpy()
        desc1 = feats1["descriptors"][0].cpu().numpy()

        return self._match_descriptors(kpts0, kpts1, desc0, desc1)

    def _match_kornia(self, path0, path1):
        _, gray0, _ = load_image(path0)
        _, gray1, _ = load_image(path1)

        tensor0 = torch.from_numpy(gray0).float()[None, None].to(device) / 255.0
        tensor1 = torch.from_numpy(gray1).float()[None, None].to(device) / 255.0

        with torch.no_grad():
            feats0 = self.aliked(tensor0)
            feats1 = self.aliked(tensor1)
            kpts0 = feats0["keypoints"][0].cpu().numpy()
            kpts1 = feats1["keypoints"][0].cpu().numpy()
            desc0 = feats0["descriptors"][0].cpu().numpy()
            desc1 = feats1["descriptors"][0].cpu().numpy()

        return self._match_descriptors(kpts0, kpts1, desc0, desc1)

    def _match_standalone(self, path0, path1):
        img0 = cv2.imread(str(path0))
        img1 = cv2.imread(str(path1))

        if max(img0.shape[:2]) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(img0.shape[:2])
            img0 = cv2.resize(img0, None, fx=scale, fy=scale)
        if max(img1.shape[:2]) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(img1.shape[:2])
            img1 = cv2.resize(img1, None, fx=scale, fy=scale)

        img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            pred0 = self.aliked.run(img0_rgb)
            pred1 = self.aliked.run(img1_rgb)
            kpts0 = pred0["keypoints"].cpu().numpy()
            kpts1 = pred1["keypoints"].cpu().numpy()
            desc0 = pred0["descriptors"].cpu().numpy()
            desc1 = pred1["descriptors"].cpu().numpy()

        return self._match_descriptors(kpts0, kpts1, desc0, desc1)

    def _match_descriptors(self, kpts0, kpts1, desc0, desc1):
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
