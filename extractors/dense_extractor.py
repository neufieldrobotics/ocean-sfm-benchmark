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
    MAX_IMAGE_DIM, LOFTR_CONFIDENCE_THRESHOLD,
    ROMA_MAX_KEYPOINTS_PER_PAIR,
    DKM_MAX_KEYPOINTS_PER_PAIR,
)


class DenseExtractor(BaseExtractor):
    """Unified extractor for LoFTR, RoMa, and DKM dense matchers."""
    is_dense = True

    # How many pairs to process before reloading the model to free leaked VRAM
    RELOAD_EVERY = {
        "loftr": 0,   # no reload needed
        "roma": 0,    # tiny fits fine
        "roma-full": 10,  # full model leaks VRAM heavily per pair
        "dkm": 50,
    }

    def __init__(self, method="loftr", device="cuda"):
        super().__init__(device)
        assert method in ("loftr", "roma", "roma-full", "dkm"), \
            f"Unsupported method: {method}."
        self.method = method
        self.name = method
        self.model = None
        self.reload_every = self.RELOAD_EVERY.get(method, 0)
        self._setup_model()

    @staticmethod
    def _nuke_cuda_tensors(obj, visited=None):
        """Recursively clear ALL CUDA tensors from a module tree,
        including unregistered attributes that .cpu() misses."""
        if visited is None:
            visited = set()
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if isinstance(obj, torch.nn.Module):
            for key in list(vars(obj).keys()):
                val = getattr(obj, key, None)
                if isinstance(val, torch.Tensor) and val.is_cuda:
                    setattr(obj, key, None)
                elif isinstance(val, (list, tuple)):
                    for item in val:
                        DenseExtractor._nuke_cuda_tensors(item, visited)
                elif isinstance(val, dict):
                    for v in val.values():
                        DenseExtractor._nuke_cuda_tensors(v, visited)
                elif isinstance(val, torch.nn.Module):
                    DenseExtractor._nuke_cuda_tensors(val, visited)
            for child in obj.children():
                DenseExtractor._nuke_cuda_tensors(child, visited)

    def unload_model(self):
        """Delete model and free all GPU memory."""
        if self.model is not None:
            # First nuke all CUDA tensors (including unregistered attrs
            # that .cpu() can't reach — roma stores activations this way)
            self._nuke_cuda_tensors(self.model)
            del self.model
            self.model = None
        import gc
        gc.collect()
        gc.collect()
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def reload_model(self):
        """Unload then reload the model to reclaim leaked VRAM."""
        print(f"    Reloading {self.name} model to free VRAM...")
        self.unload_model()
        self._setup_model()

    def _setup_model(self):
        if self.method == "loftr":
            from kornia.feature import LoFTR
            self.model = LoFTR(pretrained="outdoor").eval().to(self.device)
            print(f"LoFTR loaded on {self.device}")

        elif self.method == "roma":
            from romatch import tiny_roma_v1_outdoor
            self.model = tiny_roma_v1_outdoor(device=self.device)
            print(f"RoMa (tiny) loaded on {self.device}")

        elif self.method == "roma-full":
            from romatch import roma_outdoor
            self.model = roma_outdoor(device=self.device, upsample_res=560)
            print(f"RoMa (full, upsample_res=560) loaded on {self.device}")

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
        elif self.method in ("roma", "roma-full"):
            return self._match_roma(path0, path1, w0, h0, w1, h1)

        elif self.method == "dkm":
            return self._match_dkm(path0, path1, w0, h0, w1, h1)

    def _match_loftr(self, path0, path1):
        gray0 = cv2.imread(str(path0), cv2.IMREAD_GRAYSCALE)
        gray1 = cv2.imread(str(path1), cv2.IMREAD_GRAYSCALE)
        h0_orig, w0_orig = gray0.shape
        h1_orig, w1_orig = gray1.shape

        # Resize to MAX_IMAGE_DIM
        if max(h0_orig, w0_orig) > MAX_IMAGE_DIM:
            scale0 = MAX_IMAGE_DIM / max(h0_orig, w0_orig)
            gray0 = cv2.resize(gray0, None, fx=scale0, fy=scale0)
        else:
            scale0 = 1.0
        if max(h1_orig, w1_orig) > MAX_IMAGE_DIM:
            scale1 = MAX_IMAGE_DIM / max(h1_orig, w1_orig)
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

    @staticmethod
    def _load_roma_image_pil(path):
        """Load and resize image as PIL. Dims rounded to multiples of 14."""
        from PIL import Image
        img = Image.open(str(path)).convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(w, h)
            w, h = int(w * scale), int(h * scale)
        w = (w // 14) * 14
        h = (h // 14) * 14
        return img.resize((w, h), Image.LANCZOS)

    @staticmethod
    def _load_roma_image_tensor(path):
        """Load and resize image as tensor [1,3,H,W] in [0,1]."""
        import torchvision.transforms.functional as TF
        img = DenseExtractor._load_roma_image_pil(path)
        return TF.to_tensor(img).unsqueeze(0)

    def _match_roma(self, path0, path1, w0, h0, w1, h1):
        import gc

        # Aggressively free VRAM before each pair (critical for roma-full)
        if self.device == "cuda":
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        try:
            with torch.no_grad():
                im0 = self._load_roma_image_pil(path0)
                im1 = self._load_roma_image_pil(path1)
                warp, certainty = self.model.match(im0, im1, batched=True)
                del im0, im1

                # romatch forces batched=False for PIL input, so warp is
                # [H, W, 4]; squeeze defensively in case a tensor path returns
                # [1, H, W, 4]. sample() requires exactly three dimensions.
                if warp.dim() == 4:
                    warp, certainty = warp[0], certainty[0]

                # Use romatch's own sampler rather than a top-k over certainty.
                # sample() runs in "threshold_balanced" mode: it clamps
                # certainty above sample_thresh, draws 4N candidates by
                # multinomial, then resamples N of them weighted by inverse KDE
                # density. That inverse-density step is what spreads
                # correspondences over the whole frame. Selecting the N most
                # certain pixels instead concentrates them in a few
                # high-texture patches (measured: 22% vs 88% of a 16x16 image
                # grid occupied), which leaves the pose and triangulation
                # poorly conditioned and inflates reprojection error.
                matches, cert = self.model.sample(
                    warp, certainty, num=ROMA_MAX_KEYPOINTS_PER_PAIR)

                del warp, certainty

                if matches is None or len(matches) == 0:
                    torch.cuda.empty_cache()
                    return None, None, None

                warp_np = matches.cpu().numpy()
                cert_np = cert.cpu().numpy()
                del matches, cert

        except torch.cuda.OutOfMemoryError:
            print(f"    OOM on pair, reloading model and skipping...")
            self.reload_model()
            return None, None, None

        # Free GPU memory after each pair — RoMa is very VRAM hungry
        gc.collect()
        torch.cuda.empty_cache()

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

    @staticmethod
    def _resize_to_temp(path):
        """Load image, resize to MAX_IMAGE_DIM, save to temp file for DKM's path-based API."""
        import os, tempfile
        from PIL import Image
        img = Image.open(str(path)).convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        img.save(tmp_path)
        return tmp_path

    def _match_dkm(self, path0, path1, w0, h0, w1, h1):
        import os
        tmp0 = self._resize_to_temp(path0)
        tmp1 = self._resize_to_temp(path1)
        try:
            with torch.no_grad():
                dense_matches, dense_certainty = self.model.match(
                    tmp0, tmp1, device=self.device)
                sparse_matches, sparse_cert = self.model.sample(
                    dense_matches, dense_certainty, num=DKM_MAX_KEYPOINTS_PER_PAIR)
        finally:
            os.unlink(tmp0)
            os.unlink(tmp1)

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
