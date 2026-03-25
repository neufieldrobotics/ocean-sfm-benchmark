"""COLMAP SQLite database interface and geometry verification utilities."""

import sqlite3
import numpy as np
import cv2
from config import (
    RANSAC_REPROJ_THRESH, RANSAC_CONFIDENCE, RANSAC_MAX_TRIALS,
    MIN_INLIERS, MIN_INLIER_RATIO,
)


def array_to_blob(array: np.ndarray) -> bytes:
    return array.tobytes()


def verify_matches_cv2(kps0, kps1, matches, min_inliers=None,
                       reproj_thresh=None, confidence=None,
                       min_inlier_ratio=None):
    """Verify matches using fundamental matrix RANSAC.

    Mirrors COLMAP's TwoViewGeometry verification:
    - F-matrix RANSAC with same thresholds
    - Enforces min_inliers AND min_inlier_ratio

    Returns:
        (inlier_matches, F) or (None, None) if verification fails.
    """
    min_inliers = min_inliers if min_inliers is not None else MIN_INLIERS
    reproj_thresh = reproj_thresh if reproj_thresh is not None else RANSAC_REPROJ_THRESH
    confidence = confidence if confidence is not None else RANSAC_CONFIDENCE
    min_inlier_ratio = min_inlier_ratio if min_inlier_ratio is not None else MIN_INLIER_RATIO

    if matches is None or len(matches) < min_inliers:
        return None, None

    pts1 = kps0[matches[:, 0]]
    pts2 = kps1[matches[:, 1]]

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC,
                                     reproj_thresh, confidence)
    if F is None or mask is None:
        return None, None

    inliers = matches[mask.ravel() == 1]
    num_inliers = len(inliers)

    # Match COLMAP: enforce both min count and min ratio
    if num_inliers < min_inliers:
        return None, None
    if num_inliers / len(matches) < min_inlier_ratio:
        return None, None

    return inliers, F


class ColmapDatabase:
    """COLMAP SQLite database interface.

    Uses the ALIKED script's pair-ordering convention: when image_id1 > image_id2,
    match columns are flipped and F is transposed.
    """

    def __init__(self, database_path):
        self.connection = sqlite3.connect(database_path)
        self.create_tables()

    def create_tables(self):
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS cameras (
                camera_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                params BLOB NOT NULL,
                prior_focal_length INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS images (
                image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                camera_id INTEGER NOT NULL,
                prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
                prior_tx REAL, prior_ty REAL, prior_tz REAL,
                FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
            );
            CREATE TABLE IF NOT EXISTS keypoints (
                image_id INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB,
                FOREIGN KEY(image_id) REFERENCES images(image_id)
            );
            CREATE TABLE IF NOT EXISTS descriptors (
                image_id INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB,
                FOREIGN KEY(image_id) REFERENCES images(image_id)
            );
            CREATE TABLE IF NOT EXISTS matches (
                pair_id INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB
            );
            CREATE TABLE IF NOT EXISTS two_view_geometries (
                pair_id INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB,
                config INTEGER NOT NULL,
                F BLOB, E BLOB, H BLOB
            );
        """)
        self.connection.commit()

    def add_camera(self, width, height, model=1):
        """Add camera with SIMPLE_RADIAL model (model=1): f, cx, cy, k1."""
        focal_length = max(width, height) * 1.2
        params = np.array([focal_length, width / 2.0, height / 2.0, 0.0],
                          dtype=np.float64)
        cursor = self.connection.execute(
            "INSERT INTO cameras (model, width, height, params, prior_focal_length) "
            "VALUES (?, ?, ?, ?, ?)",
            (model, width, height, array_to_blob(params), 1),
        )
        return cursor.lastrowid

    def add_image(self, name, camera_id):
        cursor = self.connection.execute(
            "INSERT INTO images (name, camera_id) VALUES (?, ?)",
            (name, camera_id),
        )
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        keypoints = keypoints.astype(np.float32)
        self.connection.execute(
            "INSERT INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            (image_id, keypoints.shape[0], keypoints.shape[1],
             array_to_blob(keypoints)),
        )

    def add_descriptors(self, image_id, descriptors):
        descriptors = descriptors.astype(np.float32)
        self.connection.execute(
            "INSERT INTO descriptors (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            (image_id, descriptors.shape[0], descriptors.shape[1],
             array_to_blob(descriptors)),
        )

    def add_matches(self, image_id1, image_id2, matches):
        """Store matches with correct pair ordering (flips columns if needed)."""
        if image_id1 > image_id2:
            image_id1, image_id2 = image_id2, image_id1
            matches = matches[:, [1, 0]]
        pair_id = self._pair_id(image_id1, image_id2)
        self.connection.execute(
            "INSERT OR REPLACE INTO matches (pair_id, rows, cols, data) "
            "VALUES (?, ?, ?, ?)",
            (pair_id, matches.shape[0], matches.shape[1],
             array_to_blob(matches.astype(np.uint32))),
        )

    def add_two_view_geometry(self, image_id1, image_id2, matches, F,
                              E=None, H=None, config=None):
        """Store two-view geometry with correct pair ordering.

        COLMAP config values:
            0 = UNDEFINED
            1 = DEGENERATE
            2 = CALIBRATED (E-matrix computed)
            3 = UNCALIBRATED (F-matrix only)
            4 = PLANAR
        Default config=3 since we only compute F-matrix via RANSAC.
        """
        if config is None:
            # F-matrix only -> UNCALIBRATED
            config = 3
        if image_id1 > image_id2:
            image_id1, image_id2 = image_id2, image_id1
            matches = matches[:, [1, 0]]
            F = F.T
            if E is not None:
                E = E.T
        pair_id = self._pair_id(image_id1, image_id2)
        F_blob = array_to_blob(F.astype(np.float64))
        E_blob = array_to_blob((E if E is not None else np.zeros((3, 3), dtype=np.float64)).astype(np.float64))
        H_blob = array_to_blob((H if H is not None else np.zeros((3, 3), dtype=np.float64)).astype(np.float64))
        self.connection.execute(
            "INSERT OR REPLACE INTO two_view_geometries "
            "(pair_id, rows, cols, data, config, F, E, H) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id, matches.shape[0], matches.shape[1],
             array_to_blob(matches.astype(np.uint32)), config,
             F_blob, E_blob, H_blob),
        )

    @staticmethod
    def _pair_id(image_id1, image_id2):
        """image_id1 must be < image_id2."""
        return int(image_id1) * 2147483647 + int(image_id2)

    def commit(self):
        self.connection.commit()

    def close(self):
        self.connection.close()
