#!/usr/bin/env python3
"""Patch COLMAP's bundled ALIKED ONNX model to remove the 4096 feature cap.

COLMAP 4.0 ships aliked-n16rot.onnx with a hardcoded TopK k=4096 in the ONNX
graph (node /Constant_91). This limits feature extraction to 4096 keypoints
regardless of the --AlikedExtraction.max_num_features flag.

This script patches the constant from 4096 to 16384 and saves the result as
aliked-n16rot-16k.onnx, which is used by sift_native.py for uncapped extraction.

Usage:
    # Download the original model first (COLMAP auto-downloads on first run)
    python patch_aliked_onnx.py

    # The patched model is referenced in extractors/sift_native.py:
    #   --AlikedExtraction.n16rot_model_path aliked-n16rot-16k.onnx
"""

import onnx
import numpy as np

model = onnx.load("aliked-n16rot.onnx")

for node in model.graph.node:
    if node.name == "/Constant_91":
        for attr in node.attribute:
            if attr.name == "value":
                new_val = onnx.numpy_helper.from_array(
                    np.array(16384, dtype=np.int64)
                )
                attr.t.CopyFrom(new_val)
                print("Patched TopK k: 4096 -> 16384")
        break

onnx.save(model, "aliked-n16rot-16k.onnx")