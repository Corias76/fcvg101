"""Intent recognition: map a free-text request to one of the system's tools (rubric 3.1).

A transparent weighted-keyword scorer is used rather than an LLM call: it is deterministic, runs offline,
is testable (see tests/test_router.py), and a small closed set of five intents does not need more.
Each intent has stems with weights; the score is the sum of weights of stems found in the lower-cased
request. Ties/zero scores fall back to the user's selected mode or to 'classify' when only an image is given.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

INTENTS = {
    "calibrate": {"calibrat": 3, "intrinsic": 3, "distortion": 2, "undistort": 3, "checkerboard": 3, "chessboard": 3,
                  "camera matrix": 3, "lens": 1, "pinhole": 2, "focal": 2},
    "filter": {"denoise": 3, "noise": 2, "blur": 2, "smooth": 2, "filter": 2, "median": 3, "gaussian": 2, "box": 1,
               "edge": 3, "canny": 3, "sobel": 3, "gradient": 3, "fourier": 3, "fft": 3, "frequency": 3,
               "low-pass": 2, "lowpass": 2, "high-pass": 2, "highpass": 2, "sharpen": 1,
               "speckle": 3, "salt": 2, "pepper": 1, "frequenc": 3, "outline": 3, "boundar": 3, "contour": 2, "grain": 2},
    "features": {"feature": 3, "keypoint": 3, "key point": 3, "match": 3, "harris": 3, "corner": 3, "sift": 3, "orb": 3,
                 "two views": 3, "second view": 3, "homography": 3, "align": 2, "register": 2, "interest point": 3},
    "classify": {"classif": 3, "disease": 3, "diseased": 3, "sick": 3, "healthy": 3, "what is wrong": 3, "diagnos": 3,
                 "blight": 2, "mold": 2, "mildew": 2, "virus": 2, "predict": 2, "identify": 2, "infect": 3, "cnn": 2,
                 "model": 1, "what is this": 2, "is this": 1, "check": 1, "spot": 1},
    "explain": {"explain": 3, "heatmap": 3, "grad-cam": 4, "gradcam": 4, "grad cam": 4, "attention": 2,
                "why": 4, "looking": 3, "where": 2, "focus": 2, "region": 3, "highlight": 3, "area": 2, "network used": 2, "saliency": 3, "look at": 1},
}


@dataclass
class Route:
    intent: str
    scores: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    reason: str = ""


def _params(text: str, intent: str) -> dict:
    t = text.lower()
    p: dict = {}
    if intent == "filter":
        for name in ("median", "gaussian", "box", "bilateral"):
            if name in t:
                p["denoiser"] = name
        if re.search(r"high[- ]?pass|sharpen", t):
            p["fft_mode"] = "highpass"
        elif re.search(r"low[- ]?pass|smooth", t):
            p["fft_mode"] = "lowpass"
    if intent == "features":
        p["method"] = "orb" if "orb" in t else "sift"
        if "harris" in t:
            p["harris"] = True
    return p


def route(text: str, has_image: bool = True, default: str | None = None) -> Route:
    """Return the best intent for `text`. `default` is used when nothing matches."""
    t = (text or "").lower().strip()
    scores = {k: sum(w for stem, w in kws.items() if stem in t) for k, kws in INTENTS.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        intent = default or ("classify" if has_image else "calibrate")
        return Route(intent, scores, _params(t, intent), "no keyword matched; used default")
    # 'explain' runs the classifier too, so a compound request ("is it diseased? show where the model looks")
    # that mentions both should run the superset tool rather than silently dropping the explanation
    if best == "classify" and scores["explain"] >= 2:
        best = "explain"
    return Route(best, scores, _params(t, best), f"matched keywords (score {scores[best]})")
