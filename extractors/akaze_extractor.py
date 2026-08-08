"""AKAZE extractor for COLMAP pipeline.

AKAZE uses nonlinear scale spaces (Perona-Malik diffusion) which preserves
edges and boundaries better than SIFT's linear Gaussian pyramid. Uses MLDB
binary descriptor by default. Matching via BF Hamming + ratio test.
"""

import cv2
import numpy as np
from extractors.base import BaseExtractor
from config import MAX_IMAGE_DIM, MAX_MATCHES_PER_PAIR


class AKAZEExtractor(BaseExtractor):
    """AKAZE feature extractor + BF Hamming matcher for COLMAP."""
    name = "akaze"
    is_dense = False

    def __init__(self, device="cuda", max_features=MAX_MATCHES_PER_PAIR,
                 ratio_thresh=0.75):
        super().__init__(device)
        self.akaze = cv2.AKAZE_create()
        self.max_features = max_features
        self.ratio_thresh = ratio_thresh
        print(f"AKAZE extractor loaded (max_features={max_features})")

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
        kps, desc = self.akaze.detectAndCompute(gray, None)

        if kps is None or len(kps) == 0 or desc is None:
            return None

        # Cap by response (strongest first)
        if len(kps) > self.max_features:
            idx = np.argsort([k.response for k in kps])[::-1][:self.max_features]
            kps = [kps[i] for i in idx]
            desc = desc[idx]

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

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(desc0, desc1, k=2)

        good = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_thresh * n.distance:
                    good.append(m)

        if len(good) == 0:
            return None, None

        match_indices = np.array(
            [[m.queryIdx, m.trainIdx] for m in good], dtype=np.int32)
        confidences = np.array(
            [1.0 / (1.0 + m.distance) for m in good], dtype=np.float32)

        return match_indices, confidences
