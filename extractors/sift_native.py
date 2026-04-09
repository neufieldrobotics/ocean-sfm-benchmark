"""Native COLMAP extractors using separate commands for per-stage timing.

Uses COLMAP's built-in feature extractors and matchers:
- SIFT: Covariant SIFT + guided matching (same as automatic_reconstructor)
- ALIKED: ALIKED_N16ROT + LightGlue
Both use separate feature_extractor + exhaustive_matcher commands
so we get per-stage timing breakdowns.
"""

import time
import subprocess
from pathlib import Path
from config import COLMAP_BIN, MAX_IMAGE_DIM, MAX_MATCHES_PER_PAIR


class NativeColmapExtractor:
    """Runs native COLMAP feature extraction and matching.

    Uses separate commands (not automatic_reconstructor) so we get
    per-stage timing. run_colmap.py handles mapper and MVS.
    """
    is_dense = False
    is_native = True
    runs_full_pipeline = False  # we handle mapper/MVS in run_colmap.py

    def __init__(self, feature="sift"):
        assert feature in ("sift", "aliked"), \
            f"Native COLMAP supports: sift, aliked. Got: {feature}"
        self.feature = feature
        self.name = feature
        self.timings = {}

    def run(self, image_dir, database_path, **kwargs):
        """Run feature extraction + matching. Timings stored in self.timings."""
        if self.feature == "sift":
            self._run_sift(image_dir, database_path)
        elif self.feature == "aliked":
            self._run_aliked(image_dir, database_path)

    def _run_sift(self, image_dir, database_path):
        """Covariant SIFT extraction + guided exhaustive matching.

        Same settings as automatic_reconstructor quality=high:
        - estimate_affine_shape=1, domain_size_pooling=1
        - guided_matching=1
        """
        print("\n=== SIFT Feature Extraction (Covariant SIFT) ===")
        t0 = time.time()
        subprocess.run([
            COLMAP_BIN, "feature_extractor",
            "--database_path", database_path,
            "--image_path", image_dir,
            "--FeatureExtraction.max_image_size", str(MAX_IMAGE_DIM),
            "--SiftExtraction.max_num_features", "0",
            "--SiftExtraction.estimate_affine_shape", "1",
            "--SiftExtraction.domain_size_pooling", "1",
        ], check=True)
        extract_time = time.time() - t0
        print(f"  Time: {extract_time:.1f}s")

        print("\n=== SIFT Exhaustive Matching (guided) ===")
        t0 = time.time()
        subprocess.run([
            COLMAP_BIN, "exhaustive_matcher",
            "--database_path", database_path,
            "--FeatureMatching.guided_matching", "1",
        ], check=True)
        match_time = time.time() - t0
        print(f"  Time: {match_time:.1f}s")

        self.timings = {
            "feature_extraction": extract_time,
            "matching": match_time,
        }

    def _run_aliked(self, image_dir, database_path):
        """ALIKED_N16ROT extraction + LightGlue matching."""
        print("\n=== ALIKED Feature Extraction (ALIKED_N16ROT, 8192 features) ===")
        t0 = time.time()
        subprocess.run([
            COLMAP_BIN, "feature_extractor",
            "--database_path", database_path,
            "--image_path", image_dir,
            "--FeatureExtraction.type", "ALIKED_N16ROT",
            "--AlikedExtraction.min_score", "0.01",
            "--AlikedExtraction.max_num_features", "0",
            "--AlikedExtraction.n16rot_model_path", "aliked-n16rot-16k.onnx",
        ], check=True)
        extract_time = time.time() - t0
        print(f"  Time: {extract_time:.1f}s")

        print("\n=== ALIKED Exhaustive Matching (LightGlue) ===")
        t0 = time.time()
        subprocess.run([
            COLMAP_BIN, "exhaustive_matcher",
            "--database_path", database_path,
            "--FeatureMatching.type", "ALIKED_LIGHTGLUE",
        ], check=True)
        match_time = time.time() - t0
        print(f"  Time: {match_time:.1f}s")

        self.timings = {
            "feature_extraction": extract_time,
            "matching": match_time,
        }
