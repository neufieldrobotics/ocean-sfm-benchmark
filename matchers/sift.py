"""SIFT feature matcher with brute-force KNN matching."""

import cv2
import numpy as np
from typing import Tuple
from matchers.base import BaseMatcher
from config import MAX_IMAGE_DIM, MAX_MATCHES_PER_PAIR


def load_image(path):
    """Load image, resize to MAX_IMAGE_DIM if needed, return BGR, gray, scale."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray, scale


class SIFTMatcher(BaseMatcher):
    name = "SIFT"

    def __init__(self, nfeatures=MAX_MATCHES_PER_PAIR, ratio_thresh=0.75):
        self.sift = cv2.SIFT_create(nfeatures=nfeatures)
        self.ratio_thresh = ratio_thresh

    def match(self, path0, path1):
        _, gray0, _ = load_image(path0)
        _, gray1, _ = load_image(path1)

        kp0, desc0 = self.sift.detectAndCompute(gray0, None)
        kp1, desc1 = self.sift.detectAndCompute(gray1, None)

        if desc0 is None or desc1 is None:
            return np.array([]), np.array([])

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(desc0, desc1, k=2)

        good = [m for m, n in matches if m.distance < self.ratio_thresh * n.distance]

        mkpts0 = np.array([kp0[m.queryIdx].pt for m in good])
        mkpts1 = np.array([kp1[m.trainIdx].pt for m in good])
        return mkpts0, mkpts1
