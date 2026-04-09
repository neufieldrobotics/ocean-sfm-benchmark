"""SuperPoint + SuperGlue extractor for COLMAP pipeline."""

import subprocess
import sys
import numpy as np
import cv2
import torch
from pathlib import Path

from extractors.base import BaseExtractor
from config import (
    SP_MAX_DIM, SP_MAX_KEYPOINTS, SP_KEYPOINT_THRESHOLD,
    SP_NMS_RADIUS, SG_MATCH_THRESHOLD, SG_WEIGHTS, SUPERGLUE_REPO,
)


class SuperPointSuperGlueExtractor(BaseExtractor):
    name = "superpoint+superglue"

    def __init__(self, device="cuda"):
        super().__init__(device)
        self._setup_superglue()

    def _setup_superglue(self):
        superglue_path = Path("SuperGluePretrainedNetwork")
        if not superglue_path.exists():
            print("Downloading SuperGluePretrainedNetwork...")
            subprocess.run(["git", "clone", SUPERGLUE_REPO], check=True)

        if str(superglue_path) not in sys.path:
            sys.path.insert(0, str(superglue_path))

        from models.matching import Matching

        config = {
            "superpoint": {
                "nms_radius": SP_NMS_RADIUS,
                "keypoint_threshold": SP_KEYPOINT_THRESHOLD,
                "max_keypoints": SP_MAX_KEYPOINTS,
            },
            "superglue": {
                "weights": SG_WEIGHTS,
                "sinkhorn_iterations": 20,
                "match_threshold": SG_MATCH_THRESHOLD,
            },
        }
        self.matching = Matching(config).eval().to(self.device)
        print(f"SuperPoint + SuperGlue loaded on {self.device}")

    @staticmethod
    def _resize_keep_aspect(img, max_dim):
        h, w = img.shape
        if max_dim <= 0 or max(h, w) <= max_dim:
            return img, 1.0, 1.0
        scale = max_dim / max(h, w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, w / new_w, h / new_h

    def extract_features_image(self, image_path):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        h_orig, w_orig = img.shape
        img_rs, scale_x, scale_y = self._resize_keep_aspect(img, SP_MAX_DIM)
        print(f"Extracting SuperPoint features from {image_path} (original: {w_orig}x{h_orig}, resized: {img_rs.shape[1]}x{img_rs.shape[0]})")
        img_tensor = torch.from_numpy(img_rs / 255.0).float()[None, None].to(self.device)

        with torch.no_grad():
            pred = self.matching.superpoint({"image": img_tensor})

        kps_rs = pred["keypoints"][0].detach().cpu().numpy().astype(np.float32)
        desc = pred["descriptors"][0].detach().cpu().numpy().T.astype(np.float32)
        scores = pred["scores"][0].detach().cpu().numpy().astype(np.float32)

        kps_orig = kps_rs.copy()
        if scale_x != 1.0 or scale_y != 1.0:
            kps_orig[:, 0] *= scale_x
            kps_orig[:, 1] *= scale_y

        del pred
        return {
            "kps_rs": kps_rs,
            "kps_orig": kps_orig,
            "desc": desc,
            "scores": scores,
            "img_tensor": img_tensor,
            "orig_hw": (h_orig, w_orig),
        }

    def match_pair(self, feat0, feat1):
        data = {
            "image0": feat0["img_tensor"],
            "image1": feat1["img_tensor"],
            "keypoints0": torch.from_numpy(feat0["kps_rs"])[None].float().to(self.device),
            "keypoints1": torch.from_numpy(feat1["kps_rs"])[None].float().to(self.device),
            "descriptors0": torch.from_numpy(feat0["desc"].T)[None].float().to(self.device),
            "descriptors1": torch.from_numpy(feat1["desc"].T)[None].float().to(self.device),
            "scores0": torch.from_numpy(feat0["scores"])[None].float().to(self.device),
            "scores1": torch.from_numpy(feat1["scores"])[None].float().to(self.device),
        }

        with torch.no_grad():
            pred = self.matching.superglue(data)

        matches0 = pred["matches0"][0].detach().cpu().numpy()
        conf0 = pred["matching_scores0"][0].detach().cpu().numpy()

        valid = matches0 > -1
        matches_list = np.stack([np.where(valid)[0], matches0[valid]], axis=1).astype(np.int32)
        conf_valid = conf0[valid].astype(np.float32)

        del pred, data
        return matches_list, conf_valid
