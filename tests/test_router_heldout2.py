"""Second held-out set, written after the vocabulary was extended following tests/test_router_heldout.py."""
from assistant.router import route

CASES = [
    ("does this tomato plant have a fungal problem", "classify"), ("tell me the condition of this leaf", "classify"),
    ("is there any sign of mosaic virus", "classify"), ("grain removal please, the photo is noisy", "filter"),
    ("sharpen the edges", "filter"), ("give me the gradient magnitude", "filter"),
    ("blur the background with a box filter", "filter"), ("link up the same points across the two pictures", "features"),
    ("run SIFT on both images", "features"), ("extract interest points", "features"),
    ("calibrate using my chessboard photos", "calibrate"), ("what are the camera intrinsics", "calibrate"),
    ("straighten the barrel distortion", "calibrate"), ("show the heatmap for the prediction", "explain"),
    ("which part of the leaf did the model focus on", "explain"),
]
if __name__ == "__main__":
    wrong = [(t, e, route(t).intent) for t, e in CASES if route(t).intent != e]
    print(f"held-out set 2 routing accuracy: {len(CASES) - len(wrong)}/{len(CASES)}")
    for w in wrong:
        print("MISS:", w)
