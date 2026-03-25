"""Base matcher class and data structures for the benchmark."""

import time
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class MatchResult:
    method: str
    img0_name: str
    img1_name: str
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    time_taken: float
    mkpts0: np.ndarray = field(repr=False)
    mkpts1: np.ndarray = field(repr=False)
    inliers: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class BenchmarkSummary:
    method: str
    total_pairs: int
    avg_matches: float
    std_matches: float
    avg_inliers: float
    std_inliers: float
    avg_inlier_ratio: float
    avg_time: float
    total_time: float
    success_rate: float  # % of pairs with >=10 inliers


def compute_inliers(mkpts0, mkpts1, thresh=3.0):
    """Compute inliers using RANSAC with fundamental matrix."""
    if len(mkpts0) < 8:
        return np.zeros(len(mkpts0), dtype=bool), 0

    F, mask = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC,
                                     thresh, 0.999, 10000)
    if mask is None:
        return np.zeros(len(mkpts0), dtype=bool), 0

    inliers = mask.ravel().astype(bool)
    return inliers, int(inliers.sum())


def compute_summary(results, method):
    """Compute summary statistics for a method."""
    method_results = [r for r in results if r.method == method]
    if not method_results:
        return BenchmarkSummary(method, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    matches = [r.num_matches for r in method_results]
    inliers = [r.num_inliers for r in method_results]
    ratios = [r.inlier_ratio for r in method_results]
    times = [r.time_taken for r in method_results]
    successes = sum(1 for r in method_results if r.num_inliers >= 10)

    return BenchmarkSummary(
        method=method,
        total_pairs=len(method_results),
        avg_matches=np.mean(matches),
        std_matches=np.std(matches),
        avg_inliers=np.mean(inliers),
        std_inliers=np.std(inliers),
        avg_inlier_ratio=np.mean(ratios),
        avg_time=np.mean(times),
        total_time=np.sum(times),
        success_rate=successes / len(method_results) * 100,
    )


class BaseMatcher:
    name = "Base"

    def match(self, path0: str, path1: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mkpts0 [N,2], mkpts1 [N,2]) in image coordinates."""
        raise NotImplementedError

    def __call__(self, path0: str, path1: str) -> MatchResult:
        start = time.time()
        try:
            mkpts0, mkpts1 = self.match(path0, path1)
            elapsed = time.time() - start

            if len(mkpts0) == 0:
                return MatchResult(self.name, Path(path0).name, Path(path1).name,
                                   0, 0, 0.0, elapsed, np.array([]), np.array([]))

            inliers, num_inliers = compute_inliers(mkpts0, mkpts1)
            inlier_ratio = num_inliers / len(mkpts0) if len(mkpts0) > 0 else 0

            return MatchResult(self.name, Path(path0).name, Path(path1).name,
                               len(mkpts0), num_inliers, inlier_ratio, elapsed,
                               mkpts0, mkpts1, inliers)
        except Exception as e:
            print(f"    Error in {self.name}: {e}")
            elapsed = time.time() - start
            return MatchResult(self.name, Path(path0).name, Path(path1).name,
                               0, 0, 0.0, elapsed, np.array([]), np.array([]))
