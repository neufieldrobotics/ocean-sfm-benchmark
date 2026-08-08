"""Shared configuration constants for the feature benchmark."""

import torch

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# COLMAP binary
COLMAP_BIN = "colmap"

# Uniform image resolution for all detectors
MAX_IMAGE_DIM = 1600

# Uniform match cap — top matches per pair by confidence (for fair comparison)
MAX_MATCHES_PER_PAIR = 8000

# SuperPoint + SuperGlue
SP_MAX_DIM = MAX_IMAGE_DIM
SP_MAX_KEYPOINTS = MAX_MATCHES_PER_PAIR
SP_KEYPOINT_THRESHOLD = 0.0005
SP_NMS_RADIUS = 2
SG_MATCH_THRESHOLD = 0.2
SG_WEIGHTS = "outdoor"
SUPERGLUE_REPO = "https://github.com/magicleap/SuperGluePretrainedNetwork.git"

# ALIKED
ALIKED_MAX_DIM = MAX_IMAGE_DIM
ALIKED_MAX_KEYPOINTS = MAX_MATCHES_PER_PAIR
ALIKED_DETECTION_THRESHOLD = 0.001
ALIKED_NMS_RADIUS = 2

# DISK
DISK_MAX_KEYPOINTS = MAX_MATCHES_PER_PAIR  # DISK/kornia doesn't support -1 (unlimited)

# Dense matchers
ROMA_MAX_KEYPOINTS_PER_PAIR = MAX_MATCHES_PER_PAIR
# Unused by the RoMa extractor: romatch's sample() applies its own
# sample_thresh (0.05) and then inverse-density resampling, so imposing an
# extra certainty cut here would re-bias selection toward high-texture regions.
ROMA_CONFIDENCE_THRESHOLD = 0.1
LOFTR_MAX_DIM = MAX_IMAGE_DIM
LOFTR_CONFIDENCE_THRESHOLD = 0.1
DKM_MAX_KEYPOINTS_PER_PAIR = MAX_MATCHES_PER_PAIR

# Keypoint aggregation (dense matchers)
KEYPOINT_MERGE_RADIUS = 3.0

# Geometric verification — match COLMAP's TwoViewGeometry defaults exactly
RANSAC_REPROJ_THRESH = 4.0       # TwoViewGeometry.max_error
RANSAC_CONFIDENCE = 0.999        # TwoViewGeometry.confidence
RANSAC_MAX_TRIALS = 10000        # TwoViewGeometry.max_num_trials
MIN_INLIERS = 15                 # TwoViewGeometry.min_num_inliers
MIN_INLIER_RATIO = 0.25          # TwoViewGeometry.min_inlier_ratio

# VRAM management
EMPTY_CACHE_EVERY = 100

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".dng", ".bmp", ".tiff", ".tif"}

# COLMAP mapper flags — empty dict = use COLMAP's exact defaults everywhere.
# Native COLMAP defaults (for reference):
#   init_min_num_inliers = 100
#   init_max_error = 4
#   abs_pose_min_num_inliers = 30
#   abs_pose_min_inlier_ratio = 0.25
#   abs_pose_max_error = 12
#   min_num_matches = 15
#   multiple_models = 1
#   ba_refine_focal_length = 1
#   ba_refine_extra_params = 1
#   filter_max_reproj_error = 4
#   filter_min_tri_angle = 1.5
MAPPER_FLAGS = {}
