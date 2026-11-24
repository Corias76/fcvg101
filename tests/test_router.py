"""Intent-routing accuracy test.   python -m tests.test_router"""
from assistant.router import route

CASES = [
    ("Is this leaf diseased?", "classify"), ("what disease does my tomato have", "classify"),
    ("diagnose this plant", "classify"), ("is it healthy or sick", "classify"), ("classify the image", "classify"),
    ("does this look like late blight", "classify"), ("identify the infection", "classify"),
    ("denoise the picture with a median filter", "filter"), ("show me the edges", "filter"),
    ("apply canny", "filter"), ("remove noise", "filter"), ("compute the fft spectrum", "filter"),
    ("blur it with gaussian", "filter"), ("sobel gradient", "filter"), ("high-pass filter please", "filter"),
    ("match features between the two views", "features"), ("detect harris corners", "features"),
    ("find sift keypoints", "features"), ("ORB matching", "features"), ("align these two photos with a homography", "features"),
    ("calibrate the camera", "calibrate"), ("undistort this image", "calibrate"), ("what is the camera matrix", "calibrate"),
    ("use the checkerboard to get intrinsics", "calibrate"), ("estimate lens distortion", "calibrate"),
    ("show a grad-cam heatmap", "explain"), ("why did the model predict that", "explain"),
    ("explain the prediction", "explain"), ("where is the model looking", "explain"), ("saliency map", "explain"),
]

if __name__ == "__main__":
    wrong = [(t, e, route(t).intent) for t, e in CASES if route(t).intent != e]
    print(f"routing accuracy: {len(CASES) - len(wrong)}/{len(CASES)}")
    for w in wrong:
        print("MISS:", w)
