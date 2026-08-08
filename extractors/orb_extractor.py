"""ORB extractor for COLMAP pipeline.

Extracts ORB features (FAST keypoints + rotated BRIEF descriptors) per image
and matches pairs with brute-force Hamming distance + cross-check.
Results are injected into the COLMAP database via run_sparse_pipeline.

Tuned for maximum quality:
- 12 pyramid levels with scaleFactor=1.15 (finer scale sampling)
- WTA_K=3 for more discriminative binary tests
- patchSize=35 for more context per descriptor
- Cross-check matching instead of ratio test (better for binary descriptors)
"""

import cv2
import numpy as np
from extractors.base import BaseExtractor
from config import MAX_IMAGE_DIM, MAX_MATCHES_PER_PAIR


class ORBExtractor(BaseExtractor):
    """ORB feature extractor + BF Hamming cross-check matcher for COLMAP."""
    name = "orb"
    is_dense = False

    def __init__(self, device="cuda", nfeatures=MAX_MATCHES_PER_PAIR):
        super().__init__(device)
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=1.15,   # finer pyramid (default 1.2)
            nlevels=12,         # more levels (default 8)
            edgeThreshold=31,
            patchSize=35,       # larger context (default 31)
            WTA_K=3,            # 3-pixel comparisons (default 2)
        )
        # WTA_K=3 requires NORM_HAMMING2
        self.norm_type = cv2.NORM_HAMMING2
        print(f"ORB extractor loaded (nfeatures={nfeatures}, WTA_K=3, "
              f"nlevels=12, scaleFactor=1.15, patchSize=35)")

    def extract_features_image(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        h_orig, w_orig = img.shape[:2]
        scale = 1.0
        if max(h_orig, w_orig) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(h_orig, w_orig)
            img = cv2.resize(img, None, fx=scale, fy=scale)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, desc = self.orb.detectAndCompute(gray, None)

        if kps is None or len(kps) == 0 or desc is None:
            return None

        pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        scores = np.array([kp.response for kp in kps], dtype=np.float32)

        if scale != 1.0:
            pts_orig = pts / scale
        else:
            pts_orig = pts.copy()

        desc_float = desc.astype(np.float32)

        return {
            "kps_orig": pts_orig,
            "desc": desc_float,
            "desc_binary": desc,
            "scores": scores,
            "orig_hw": (h_orig, w_orig),
        }

    def match_pair(self, feat0, feat1):
        desc0 = feat0["desc_binary"]
        desc1 = feat1["desc_binary"]

        if desc0 is None or desc1 is None or len(desc0) < 2 or len(desc1) < 2:
            return None, None

        # Cross-check: each match must be best in both directions
        bf = cv2.BFMatcher(self.norm_type, crossCheck=True)
        matches = bf.match(desc0, desc1)

        if len(matches) == 0:
            return None, None

        # Sort by distance (best first)
        matches = sorted(matches, key=lambda m: m.distance)

        match_indices = np.array(
            [[m.queryIdx, m.trainIdx] for m in matches], dtype=np.int32)
        confidences = np.array(
            [1.0 / (1.0 + m.distance) for m in matches], dtype=np.float32)

        return match_indices, confidences
