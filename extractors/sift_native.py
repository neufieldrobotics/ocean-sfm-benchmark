"""Native COLMAP SIFT extractor.

Matches colmap automatic_reconstructor behavior exactly:
- Covariant SIFT (affine shape + domain size pooling)
- Guided matching (uses geometry to find additional matches)
"""

import subprocess
from config import COLMAP_BIN


class SIFTNativeExtractor:
    """Wrapper for native COLMAP SIFT pipeline.

    Replicates automatic_reconstructor settings:
        SiftExtraction.estimate_affine_shape = 1
        SiftExtraction.domain_size_pooling = 1
        SiftMatching.guided_matching = 1
    """
    name = "sift"
    is_dense = False
    is_native = True

    def run(self, image_dir, database_path):
        """Run COLMAP's native SIFT extraction and exhaustive matching."""
        print("\n=== SIFT Feature Extraction (Covariant SIFT) ===")
        subprocess.run([
            COLMAP_BIN, "feature_extractor",
            "--database_path", database_path,
            "--image_path", image_dir,
            "--SiftExtraction.estimate_affine_shape", "1",
            "--SiftExtraction.domain_size_pooling", "1",
        ], check=True)

        print("\n=== SIFT Exhaustive Matching (guided) ===")
        subprocess.run([
            COLMAP_BIN, "exhaustive_matcher",
            "--database_path", database_path,
            "--FeatureMatching.guided_matching", "1",
        ], check=True)
