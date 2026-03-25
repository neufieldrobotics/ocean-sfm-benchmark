"""Unified dense extractor for LoFTR, RoMa, DKM.

Dense matchers produce per-pair correspondences (no per-image keypoints).
They require the KeypointAggregator for COLMAP integration.
"""

import numpy as np
import cv2
import torch
from pathlib import Path

from extractors.base import BaseExtractor
from config import (
    LOFTR_MAX_DIM, LOFTR_CONFIDENCE_THRESHOLD,
    ROMA_MAX_KEYPOINTS_PER_PAIR, ROMA_CONFIDENCE_THRESHOLD,
    DKM_MAX_KEYPOINTS_PER_PAIR,
)


class DenseExtractor(BaseExtractor):
    """Unified extractor for LoFTR, RoMa, and DKM dense matchers."""
    is_dense = True

    def __init__(self, method="loftr", device="cuda"):
        super().__init__(device)
        assert method in ("loftr", "roma", "dkm"), f"Unsupported method: {method}"
        self.method = method
        self.name = method
        self.model = None
        self._setup_model()

    def _setup_model(self):
        if self.method == "loftr":
            from kornia.feature import LoFTR
            self.model = LoFTR(pretrained="outdoor").eval().to(self.device)
            print(f"LoFTR loaded on {self.device}")

        elif self.method == "roma":
            from romatch import roma_outdoor
            self.model = roma_outdoor(device=self.device)
            print(f"RoMa loaded on {self.device}")

        elif self.method == "dkm":
            loaded = False
            try:
                from dkm import DKMv3_outdoor
                self.model = DKMv3_outdoor(device=self.device)
                loaded = True
            except (ImportError, AttributeError):
                pass
            if not loaded:
                try:
                    from dkm import DKMv3
                    self.model = DKMv3(device=self.device, outdoor=True)
                    loaded = True
                except (ImportError, AttributeError):
                    pass
            if not loaded:
                try:
                    from dkm import dkm_base
                    self.model = dkm_base(device=self.device)
                    loaded = True
                except (ImportError, AttributeError):
                    pass
            if loaded:
                print(f"DKM loaded on {self.device}")
            else:
                raise ImportError("No compatible DKM API found")

    def extract_features_image(self, image_path):
        """Dense matchers don't extract per-image features; just record dims."""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        return {"orig_hw": (h, w), "image_path": image_path}

    def match_pair(self, feat0, feat1):
        """Run dense matching. Returns (pts0 [M,2], pts1 [M,2], conf [M])."""
        path0 = feat0["image_path"]
        path1 = feat1["image_path"]
        h0, w0 = feat0["orig_hw"]
        h1, w1 = feat1["orig_hw"]

        if self.method == "loftr":
            return self._match_loftr(path0, path1)
        elif self.method == "roma":
            return self._match_roma(path0, path1, w0, h0, w1, h1)
        elif self.method == "dkm":
            return self._match_dkm(path0, path1, w0, h0, w1, h1)

    def _match_loftr(self, path0, path1):
        gray0 = cv2.imread(str(path0), cv2.IMREAD_GRAYSCALE)
        gray1 = cv2.imread(str(path1), cv2.IMREAD_GRAYSCALE)
        h0_orig, w0_orig = gray0.shape
        h1_orig, w1_orig = gray1.shape

        # Resize to max dim
        max_dim = LOFTR_MAX_DIM
        if max(h0_orig, w0_orig) > max_dim:
            scale0 = max_dim / max(h0_orig, w0_orig)
            gray0 = cv2.resize(gray0, None, fx=scale0, fy=scale0)
        else:
            scale0 = 1.0
        if max(h1_orig, w1_orig) > max_dim:
            scale1 = max_dim / max(h1_orig, w1_orig)
            gray1 = cv2.resize(gray1, None, fx=scale1, fy=scale1)
        else:
            scale1 = 1.0

        # LoFTR requires dimensions divisible by 8
        h0, w0 = (gray0.shape[0] // 8) * 8, (gray0.shape[1] // 8) * 8
        h1, w1 = (gray1.shape[0] // 8) * 8, (gray1.shape[1] // 8) * 8
        gray0, gray1 = gray0[:h0, :w0], gray1[:h1, :w1]

        tensor0 = torch.from_numpy(gray0).float()[None, None].to(self.device) / 255.0
        tensor1 = torch.from_numpy(gray1).float()[None, None].to(self.device) / 255.0

        with torch.no_grad():
            result = self.model({"image0": tensor0, "image1": tensor1})

        mkpts0 = result["keypoints0"].cpu().numpy()
        mkpts1 = result["keypoints1"].cpu().numpy()
        conf = result["confidence"].cpu().numpy()

        mask = conf > LOFTR_CONFIDENCE_THRESHOLD
        mkpts0, mkpts1, conf = mkpts0[mask], mkpts1[mask], conf[mask]

        if len(mkpts0) == 0:
            return None, None, None

        # Scale back to original coordinates
        mkpts0[:, 0] /= scale0
        mkpts0[:, 1] /= scale0
        mkpts1[:, 0] /= scale1
        mkpts1[:, 1] /= scale1

        return mkpts0.astype(np.float32), mkpts1.astype(np.float32), conf

    def _match_roma(self, path0, path1, w0, h0, w1, h1):
        with torch.no_grad():
            warp, certainty = self.model.match(str(path0), str(path1),
                                               device=self.device)

        warp_flat = warp.reshape(-1, 4)
        cert_flat = certainty.reshape(-1)

        mask = cert_flat > ROMA_CONFIDENCE_THRESHOLD
        warp_good = warp_flat[mask]
        cert_good = cert_flat[mask]

        if len(warp_good) == 0:
            return None, None, None

        if len(warp_good) > ROMA_MAX_KEYPOINTS_PER_PAIR:
            topk_idx = torch.topk(cert_good, ROMA_MAX_KEYPOINTS_PER_PAIR).indices
            warp_good = warp_good[topk_idx]
            cert_good = cert_good[topk_idx]

        warp_np = warp_good.cpu().numpy()
        cert_np = cert_good.cpu().numpy()

        pts0 = np.stack([
            (warp_np[:, 0] + 1) / 2 * w0,
            (warp_np[:, 1] + 1) / 2 * h0,
        ], axis=1).astype(np.float32)

        pts1 = np.stack([
            (warp_np[:, 2] + 1) / 2 * w1,
            (warp_np[:, 3] + 1) / 2 * h1,
        ], axis=1).astype(np.float32)

        # Bounds check
        valid = (
            (pts0[:, 0] >= 0) & (pts0[:, 0] < w0) &
            (pts0[:, 1] >= 0) & (pts0[:, 1] < h0) &
            (pts1[:, 0] >= 0) & (pts1[:, 0] < w1) &
            (pts1[:, 1] >= 0) & (pts1[:, 1] < h1)
        )
        return pts0[valid], pts1[valid], cert_np[valid] if isinstance(cert_np, np.ndarray) else cert_np

    def _match_dkm(self, path0, path1, w0, h0, w1, h1):
        with torch.no_grad():
            dense_matches, dense_certainty = self.model.match(
                str(path0), str(path1), device=self.device)
            sparse_matches, sparse_cert = self.model.sample(
                dense_matches, dense_certainty, num=DKM_MAX_KEYPOINTS_PER_PAIR)

        mkpts0 = sparse_matches[:, :2].cpu().numpy()
        mkpts1 = sparse_matches[:, 2:].cpu().numpy()

        mkpts0[:, 0] = (mkpts0[:, 0] + 1) / 2 * w0
        mkpts0[:, 1] = (mkpts0[:, 1] + 1) / 2 * h0
        mkpts1[:, 0] = (mkpts1[:, 0] + 1) / 2 * w1
        mkpts1[:, 1] = (mkpts1[:, 1] + 1) / 2 * h1

        if hasattr(sparse_cert, 'cpu'):
            cert = sparse_cert.cpu().numpy()
        else:
            cert = np.ones(len(mkpts0), dtype=np.float32)

        return mkpts0.astype(np.float32), mkpts1.astype(np.float32), cert
