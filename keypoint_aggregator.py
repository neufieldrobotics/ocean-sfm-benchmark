"""Keypoint aggregation for dense matchers (LoFTR, RoMa, DKM).

Dense matchers produce per-pair pixel correspondences with no consistent
keypoint IDs. This module clusters observations across all pairs into
canonical per-image keypoints and re-indexes matches to use those IDs.

Pair matches are streamed to HDF5 (float16 + gzip) during matching to
avoid holding all pairs in memory. During aggregation, points are read
back one image at a time, keeping peak memory proportional to the
largest single image's observations rather than all pairs combined.
"""

import os
import tempfile
import numpy as np
import h5py
from scipy.spatial import cKDTree
from collections import defaultdict
from tqdm import tqdm
from config import KEYPOINT_MERGE_RADIUS


class KeypointAggregator:
    """Collect pixel locations from all pair matches, cluster per image,
    and build consistent keypoint sets with re-indexed matches.

    Pair data is streamed to a temporary HDF5 file (float16, gzip
    compressed) so that only one image's points need to be in memory
    at a time during aggregation.
    """

    def __init__(self, merge_radius=KEYPOINT_MERGE_RADIUS, h5_path=None,
                 read_only=False):
        """
        Args:
            merge_radius: greedy clustering radius in pixels.
            h5_path: where to stream pair correspondences. If None a temp file
                is created and deleted on close(). If given, the file is kept,
                which lets an expensive dense matching pass be reused.
            read_only: open an EXISTING correspondence file instead of creating
                one. Used by the merge-radius ablation to re-aggregate a single
                cached matching run at many radii without re-running the matcher.
        """
        self.merge_radius = merge_radius
        self._pair_count = 0
        self.read_only = read_only

        if read_only:
            if h5_path is None:
                raise ValueError("read_only=True requires an existing h5_path")
            self._owns_h5 = False
            self.h5_path = h5_path
            self.h5 = h5py.File(h5_path, "r")
            self._pair_count = len(self.h5["pairs"])
            return

        if h5_path is None:
            fd, h5_path = tempfile.mkstemp(suffix=".h5")
            os.close(fd)
            self._owns_h5 = True
        else:
            self._owns_h5 = False

        self.h5_path = h5_path
        self.h5 = h5py.File(h5_path, "w")
        self.h5.create_group("pairs")

    # ------------------------------------------------------------------
    # Phase 1: stream pairs to disk
    # ------------------------------------------------------------------

    def add_pair(self, img0_name, img1_name, pts0, pts1, confidences):
        """Write one pair's dense matches to HDF5 (float16 + gzip)."""
        grp = self.h5["pairs"].create_group(f"{self._pair_count:06d}")
        grp.attrs["img0"] = img0_name
        grp.attrs["img1"] = img1_name
        grp.create_dataset("pts0", data=pts0.astype(np.float32),
                           compression="gzip", compression_opts=1)
        grp.create_dataset("pts1", data=pts1.astype(np.float32),
                           compression="gzip", compression_opts=1)
        if confidences is not None and hasattr(confidences, "astype"):
            grp.create_dataset("confs", data=confidences.astype(np.float32),
                               compression="gzip", compression_opts=1)
        self._pair_count += 1

    # ------------------------------------------------------------------
    # Phase 2: aggregate — one image at a time
    # ------------------------------------------------------------------

    def aggregate(self):
        """Cluster all collected points per image into canonical keypoints.

        Reads from HDF5 per-image so only one image's observations are
        in memory at a time.

        Returns:
            keypoints_per_image: {img_name: np.ndarray (K, 2)}
            kd_trees:            {img_name: cKDTree}
        """
        if not self.read_only:
            self.h5.flush()

        # Build lightweight index: image_name -> [(pair_key, is_img0)]
        image_pair_index = defaultdict(list)
        for pair_key in self.h5["pairs"]:
            grp = self.h5["pairs"][pair_key]
            image_pair_index[grp.attrs["img0"]].append((pair_key, True))
            image_pair_index[grp.attrs["img1"]].append((pair_key, False))

        keypoints_per_image = {}
        kd_trees = {}

        for img_name in tqdm(sorted(image_pair_index.keys()),
                             desc="Aggregating keypoints"):
            # Read only this image's points from relevant pairs
            chunks = []
            for pair_key, is_img0 in image_pair_index[img_name]:
                ds = "pts0" if is_img0 else "pts1"
                pts = self.h5["pairs"][pair_key][ds][:].astype(np.float32)
                chunks.append(pts)

            if not chunks:
                keypoints_per_image[img_name] = np.zeros((0, 2), dtype=np.float32)
                continue

            all_pts = np.concatenate(chunks, axis=0)
            del chunks

            canonical = self._cluster_points(all_pts)
            del all_pts

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

    # ------------------------------------------------------------------
    # Phase 3: re-index matches against canonical keypoints
    # ------------------------------------------------------------------

    def reindex_matches(self, kd_trees):
        """Map raw pixel matches to nearest canonical keypoint indices.

        Reads pairs from HDF5 one at a time.

        Returns:
            List of (img0_name, img1_name, matches [M, 2], confidences)
        """
        reindexed = []

        for pair_key in tqdm(sorted(self.h5["pairs"].keys()),
                             desc="Re-indexing matches"):
            grp = self.h5["pairs"][pair_key]
            img0 = grp.attrs["img0"]
            img1 = grp.attrs["img1"]

            tree0 = kd_trees.get(img0)
            tree1 = kd_trees.get(img1)
            if tree0 is None or tree1 is None:
                continue

            pts0 = grp["pts0"][:].astype(np.float32)
            pts1 = grp["pts1"][:].astype(np.float32)
            confs = grp["confs"][:] if "confs" in grp else None

            _, idx0 = tree0.query(pts0)
            _, idx1 = tree1.query(pts1)

            matches = np.stack([idx0, idx1], axis=1).astype(np.int32)
            unique_matches = np.unique(matches, axis=0)
            reindexed.append((img0, img1, unique_matches, confs))

        return reindexed

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Close HDF5 file and remove temp file if we created it."""
        try:
            self.h5.close()
        except Exception:
            pass
        if self._owns_h5:
            try:
                os.unlink(self.h5_path)
            except OSError:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
