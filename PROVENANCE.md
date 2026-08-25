# Provenance

Every number in the paper was produced by the code at the tag
**`paper-runs`**, not by the tip of `main`.

```bash
git checkout paper-runs
```

## Why the distinction matters

All benchmark databases were written between 2026-04-01 and 2026-04-20. Three
commits after `paper-runs` change how RoMa selects correspondences from its
dense warp field:

| Commit | Change |
|---|---|
| `RoMa: use romatch sample() instead of top-k over certainty` | Replaces top-k selection with inverse-density balanced sampling |
| `RoMa: keep a certainty floor alongside inverse-density balancing` | Restores a confidence threshold under the new sampler |
| `RoMa: don't starve wide-baseline pairs when balancing correspondences` | Removes a cap that discarded most correspondences on wide-baseline pairs |

At `paper-runs`, RoMa keeps the 8000 most confident correspondences per pair.
That selection concentrates them in a few high-texture patches — measured at
22% of a 16x16 image grid, against 88% after the change — which leaves pose and
triangulation poorly conditioned. We believe this contributes to the pose
inconsistency reported for RoMa in the paper (28% of adjacent camera pairs
disagreeing with the cross-method consensus on the hydrothermal sequence).

**The paper does not report results from the post-`paper-runs` code.** Those
runs exist but were kept out so that every reported number comes from one
internally consistent set of runs. `results/pose_plausibility.json` includes
them under the key `RoMa*fixed` for reference only.

If you check out `main` and rerun the benchmark, the RoMa rows will differ from
the paper. Every other method is unaffected.

## Dataset naming

Directory names in the code predate the paper's terminology:

| Directory | Paper |
|---|---|
| `MVS` | Glacier (Svalbard) |
| `MVS-HyrdoThermal` | Hydrothermal vent (Bio9, East Pacific Rise) |
| `MVS-cityhall` | City Hall (Westmount) |

The spelling `HyrdoThermal` is a typo preserved so that the released result
files match the paths recorded during the runs. Likewise `MVS-cityhall` is
Westmount City Hall (`6-Westmount-City-Hall.zip` in Heritage3DMTL, images
DJI_0892--0957), not the Montreal City Hall capture in the same dataset.
