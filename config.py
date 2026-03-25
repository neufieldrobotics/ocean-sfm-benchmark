"""Shared configuration constants for the feature benchmark."""

import torch

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# COLMAP binary
COLMAP_BIN = "colmap"

# SuperPoint + SuperGlue
SP_MAX_DIM = 1600
SP_MAX_KEYPOINTS = -1
SP_KEYPOINT_THRESHOLD = 0.05
SP_NMS_RADIUS = 3
SG_MATCH_THRESHOLD = 0.3
SG_WEIGHTS = "outdoor"
SUPERGLUE_REPO = "https://github.com/magicleap/SuperGluePretrainedNetwork.git"

# ALIKED
ALIKED_MAX_DIM = 3200
ALIKED_MAX_KEYPOINTS = -1
ALIKED_DETECTION_THRESHOLD = 0.2
ALIKED_NMS_RADIUS = 1

# DISK
DISK_MAX_KEYPOINTS = 4096

# Dense matchers
ROMA_MAX_KEYPOINTS_PER_PAIR = 2048
ROMA_CONFIDENCE_THRESHOLD = 0.1
LOFTR_MAX_DIM = 840
LOFTR_CONFIDENCE_THRESHOLD = 0.3
DKM_MAX_KEYPOINTS_PER_PAIR = 2048

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
