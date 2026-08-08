"""Benchmark matcher registry."""

from matchers.sift import SIFTMatcher
from matchers.superglue import SuperGlueMatcher
from matchers.loftr import LoFTRMatcher
from matchers.aliked import ALIKEDMatcher
from matchers.roma import RoMaMatcher
from matchers.dkm import DKMMatcher
from matchers.lightglue_matcher import LightGlueMatcher
from matchers.disk_bf import DISKBFMatcher
from matchers.orb import ORBMatcher
from matchers.akaze import AKAZEMatcher


# Registry: name -> factory function
AVAILABLE_MATCHERS = {
    "sift": lambda: SIFTMatcher(),
    "orb": lambda: ORBMatcher(),
    "akaze": lambda: AKAZEMatcher(),
    "superglue": lambda: SuperGlueMatcher(),
    "loftr": lambda: LoFTRMatcher(),
    "aliked": lambda: ALIKEDMatcher(),
    "aliked+lg": lambda: LightGlueMatcher("aliked"),
    "sp+lg": lambda: LightGlueMatcher("superpoint"),
    "disk": lambda: DISKBFMatcher(),
    "disk+lg": lambda: LightGlueMatcher("disk"),
    "roma": lambda: RoMaMatcher(),
    "roma-full": lambda: RoMaMatcher(variant="full"),
    "dkm": lambda: DKMMatcher(),
}

MATCHER_NAMES = list(AVAILABLE_MATCHERS.keys())


def init_matchers(names=None):
    """Initialize matchers by name. Returns list of (name, matcher) for
    matchers that initialized successfully."""
    if names is None:
        names = MATCHER_NAMES

    active = []
    for name in names:
        if name not in AVAILABLE_MATCHERS:
            print(f"  Unknown matcher: {name}, skipping.")
            continue
        print(f"  - {name}")
        matcher = AVAILABLE_MATCHERS[name]()
        # Check if matcher has a model loaded
        has_model = True
        if hasattr(matcher, "matching") and matcher.matching is None:
            has_model = False
        if hasattr(matcher, "loftr") and matcher.loftr is None:
            has_model = False
        if hasattr(matcher, "aliked") and matcher.aliked is None:
            has_model = False
        if hasattr(matcher, "roma") and matcher.roma is None:
            has_model = False
        if hasattr(matcher, "dkm") and matcher.dkm is None:
            has_model = False
        if hasattr(matcher, "extractor") and matcher.extractor is None:
            has_model = False

        if has_model:
            active.append(matcher)
        else:
            print(f"    (skipped - model not available)")

    return active
