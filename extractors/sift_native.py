"""Native COLMAP extractors.

Uses COLMAP's built-in feature extractors and matchers:
- SIFT: via automatic_reconstructor (Covariant SIFT + guided matching)
- ALIKED: via separate commands (feature_extractor + exhaustive_matcher)
  since automatic_reconstructor doesn't expose AlikedExtraction flags.
"""

import os
import subprocess
import shutil
from pathlib import Path
from config import COLMAP_BIN


class NativeColmapExtractor:
    """Runs native COLMAP feature extraction, matching, and reconstruction.

    For SIFT: uses automatic_reconstructor (handles full pipeline).
    For ALIKED: uses separate commands with configurable max_num_features.
    """
    is_dense = False
    is_native = True

    def __init__(self, feature="sift"):
        assert feature in ("sift", "aliked"), \
            f"Native COLMAP supports: sift, aliked. Got: {feature}"
        self.feature = feature
        self.name = feature
        # SIFT uses automatic_reconstructor (full pipeline)
        # ALIKED uses separate commands (needs mapper/MVS from run_colmap.py)
        self.runs_full_pipeline = (feature == "sift")

    def run(self, image_dir, output_dir, quality="high", skip_dense=False):
        """Run the appropriate native COLMAP pipeline."""
        if self.feature == "sift":
            self._run_automatic(image_dir, output_dir, quality, skip_dense)
        elif self.feature == "aliked":
            self._run_aliked(image_dir, output_dir)

    def _run_automatic(self, image_dir, output_dir, quality, skip_dense):
        """Run colmap automatic_reconstructor for SIFT."""
        image_dir = str(Path(image_dir).resolve())
        output_dir = str(Path(output_dir).resolve())
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            COLMAP_BIN, "automatic_reconstructor",
            "--workspace_path", output_dir,
            "--image_path", image_dir,
            "--quality", quality,
            "--feature", self.feature,
        ]
        if skip_dense:
            cmd.extend(["--dense", "0"])

        print(f"\n=== COLMAP automatic_reconstructor "
              f"(feature={self.feature}, quality={quality}) ===")
        subprocess.run(cmd, check=True)

    def _run_aliked(self, image_dir, database_path):
        """Run ALIKED extraction + matching via separate COLMAP commands.

        Uses ALIKED_N16ROT extractor and ALIKED_LIGHTGLUE matcher with
        increased feature count (8192) to match SIFT's feature budget.
        """
        print("\n=== ALIKED Feature Extraction (ALIKED_N16ROT, 8192 features) ===")
        subprocess.run([
            COLMAP_BIN, "feature_extractor",
            "--database_path", database_path,
            "--image_path", image_dir,
            "--FeatureExtraction.type", "ALIKED_N16ROT",
            "--AlikedExtraction.max_num_features", "8192",
        ], check=True)

        print("\n=== ALIKED Exhaustive Matching (LightGlue) ===")
        subprocess.run([
            COLMAP_BIN, "exhaustive_matcher",
            "--database_path", database_path,
            "--FeatureMatching.type", "ALIKED_LIGHTGLUE",
        ], check=True)
