"""Fresh phrasings written AFTER tuning the router on tests/test_router.py (held-out check)."""
from assistant.router import route

CASES = [
    ("my tomato leaves have brown spots, what is wrong", "classify"), ("can you tell if the plant is infected", "classify"),
    ("run the neural network on this photo", "classify"), ("remove the salt and pepper speckles", "filter"),
    ("which frequencies dominate this image", "filter"), ("outline the leaf boundary", "filter"),
    ("find matching points in both pictures", "features"), ("detect corners in the leaf", "features"),
    ("compare keypoints from the two shots", "features"), ("get the focal length of my camera", "calibrate"),
    ("fix the lens warping", "calibrate"), ("rectify the fisheye distortion", "calibrate"),
    ("what regions made you say that", "explain"), ("highlight the area the network used", "explain"),
    ("tell me why it thinks this is mold", "explain"),
]
if __name__ == "__main__":
    wrong = [(t, e, route(t).intent) for t, e in CASES if route(t).intent != e]
    print(f"held-out routing accuracy: {len(CASES) - len(wrong)}/{len(CASES)}")
    for w in wrong:
        print("MISS:", w)
