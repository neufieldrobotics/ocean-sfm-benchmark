"""Keypoint aggregation for dense matchers (LoFTR, RoMa, DKM).

Dense matchers produce per-pair pixel correspondences with no consistent
keypoint IDs. This module clusters observations across all pairs into
canonical per-image keypoints and re-indexes matches to use those IDs.
"""

import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict
from tqdm import tqdm
from config import KEYPOINT_MERGE_RADIUS


class KeypointAggregator:
    """Collect pixel locations from all pair matches, cluster per image,
    and build consistent keypoint sets with re-indexed matches."""

    def __init__(self, merge_radius=KEYPOINT_MERGE_RADIUS):
        self.merge_radius = merge_radius
        self.image_points = defaultdict(list)
        self.raw_pairs = []

    def add_pair(self, img0_name, img1_name, pts0, pts1, confidences):
        """Register a pair's dense matches for later aggregation."""
        self.image_points[img0_name].extend(pts0.tolist())
        self.image_points[img1_name].extend(pts1.tolist())
        self.raw_pairs.append((img0_name, img1_name, pts0, pts1, confidences))

    def aggregate(self):
        """Cluster all collected points per image into canonical keypoints.

        Returns:
            keypoints_per_image: {img_name: np.ndarray (K, 2)}
            kd_trees: {img_name: cKDTree}
        """
        keypoints_per_image = {}
        kd_trees = {}

        for img_name, points in tqdm(self.image_points.items(),
                                     desc="Aggregating keypoints"):
            pts = np.array(points, dtype=np.float32)
            if len(pts) == 0:
                keypoints_per_image[img_name] = np.zeros((0, 2), dtype=np.float32)
                continue

            canonical = self._cluster_points(pts)
            keypoints_per_image[img_name] = canonical
            kd_trees[img_name] = cKDTree(canonical)

        return keypoints_per_image, kd_trees

    def _cluster_points(self, pts):
        """Greedy radius clustering: merge nearby points into centroids."""
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        tree = cKDTree(pts)
        used = np.zeros(len(pts), dtype=bool)
        canonical = []

        for i in range(len(pts)):
            if used[i]:
                continue
            neighbors = tree.query_ball_point(pts[i], self.merge_radius)
            centroid = pts[neighbors].mean(axis=0)
            canonical.append(centroid)
            used[neighbors] = True

        return np.array(canonical, dtype=np.float32)

    def reindex_matches(self, kd_trees):
        """Map raw pixel matches to nearest canonical keypoint indices.

        Returns:
            List of (img0_name, img1_name, matches [M, 2], confidences)
        """
        reindexed = []

        for img0_name, img1_name, pts0, pts1, confs in tqdm(
                self.raw_pairs, desc="Re-indexing matches"):
            tree0 = kd_trees.get(img0_name)
            tree1 = kd_trees.get(img1_name)
            if tree0 is None or tree1 is None:
                continue

            _, idx0 = tree0.query(pts0)
            _, idx1 = tree1.query(pts1)

            matches = np.stack([idx0, idx1], axis=1).astype(np.int32)
            unique_matches = np.unique(matches, axis=0)
            reindexed.append((img0_name, img1_name, unique_matches, confs))

        return reindexed
