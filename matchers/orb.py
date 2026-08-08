"""ORB feature matcher with brute-force cross-check matching.

Tuned ORB: finer pyramid (12 levels, 1.15 scale), WTA_K=3,
larger patches (35px), cross-check matching.
"""

import cv2
import numpy as np
from matchers.base import BaseMatcher
from matchers.sift import load_image
from config import MAX_MATCHES_PER_PAIR


class ORBMatcher(BaseMatcher):
    name = "ORB"

    def __init__(self, nfeatures=MAX_MATCHES_PER_PAIR):
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=1.15,
            nlevels=12,
            edgeThreshold=31,
            patchSize=35,
            WTA_K=3,
        )
        self.norm_type = cv2.NORM_HAMMING2

    def match(self, path0, path1):
        _, gray0, _ = load_image(path0)
        _, gray1, _ = load_image(path1)

        kp0, desc0 = self.orb.detectAndCompute(gray0, None)
        kp1, desc1 = self.orb.detectAndCompute(gray1, None)

        if desc0 is None or desc1 is None or len(desc0) < 2 or len(desc1) < 2:
            return np.array([]), np.array([])

        bf = cv2.BFMatcher(self.norm_type, crossCheck=True)
        matches = bf.match(desc0, desc1)

        if not matches:
            return np.array([]), np.array([])

        matches = sorted(matches, key=lambda m: m.distance)

        mkpts0 = np.array([kp0[m.queryIdx].pt for m in matches])
        mkpts1 = np.array([kp1[m.trainIdx].pt for m in matches])
        return mkpts0, mkpts1
