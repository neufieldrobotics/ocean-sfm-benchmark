"""AKAZE feature matcher with brute-force KNN matching.

AKAZE uses nonlinear scale spaces (vs SIFT's linear Gaussian) which
preserves edges better. Uses MLDB binary descriptor by default.
"""

import cv2
import numpy as np
from matchers.base import BaseMatcher
from matchers.sift import load_image
from config import MAX_MATCHES_PER_PAIR


class AKAZEMatcher(BaseMatcher):
    name = "AKAZE"

    def __init__(self, max_features=MAX_MATCHES_PER_PAIR, ratio_thresh=0.75):
        self.akaze = cv2.AKAZE_create()
        self.max_features = max_features
        self.ratio_thresh = ratio_thresh

    def match(self, path0, path1):
        _, gray0, _ = load_image(path0)
        _, gray1, _ = load_image(path1)

        kp0, desc0 = self.akaze.detectAndCompute(gray0, None)
        kp1, desc1 = self.akaze.detectAndCompute(gray1, None)

        if desc0 is None or desc1 is None or len(desc0) < 2 or len(desc1) < 2:
            return np.array([]), np.array([])

        # Cap features by response (strongest first)
        if len(kp0) > self.max_features:
            idx = np.argsort([k.response for k in kp0])[::-1][:self.max_features]
            kp0 = [kp0[i] for i in idx]
            desc0 = desc0[idx]
        if len(kp1) > self.max_features:
            idx = np.argsort([k.response for k in kp1])[::-1][:self.max_features]
            kp1 = [kp1[i] for i in idx]
            desc1 = desc1[idx]

        # AKAZE default is MLDB (binary) -> NORM_HAMMING
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(desc0, desc1, k=2)

        good = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_thresh * n.distance:
                    good.append(m)

        if not good:
            return np.array([]), np.array([])

        mkpts0 = np.array([kp0[m.queryIdx].pt for m in good])
        mkpts1 = np.array([kp1[m.trainIdx].pt for m in good])
        return mkpts0, mkpts1
