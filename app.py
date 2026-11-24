"""Task 3: Streamlit assistant.   streamlit run app.py

Flow: user uploads an image and types a request (or picks a mode) -> router identifies the intent
-> the matching tool runs the Task 1 pipeline or Task 2 model -> visual + numeric results are shown.
"""
import cv2
import numpy as np
import streamlit as st

from assistant.router import INTENTS, route
from assistant.tools import TOOLS

st.set_page_config(page_title="Leaf Inspection Assistant", layout="wide")
st.title("🍅 Leaf Inspection Assistant")
st.caption("Computer-vision assistant for an organic tomato farm: calibrate the camera, filter, match features, "
           "and diagnose leaf disease with a CNN.")


def read(upload):
    return cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR) if upload else None


with st.sidebar:
    st.header("Input")
    up = st.file_uploader("Leaf / scene image", type=["jpg", "jpeg", "png"])
    up2 = st.file_uploader("Second view (optional, for feature matching)", type=["jpg", "jpeg", "png"])
    calib_ups = st.file_uploader("Chessboard images (optional, for calibration)", type=["jpg", "jpeg", "png"],
                                 accept_multiple_files=True)
    mode = st.selectbox("Mode", ["Auto (from my request)"] + list(INTENTS))
    st.caption("Auto mode reads your request text; picking a mode overrides it.")

request = st.text_input("What would you like to do?", placeholder="e.g. 'Is this leaf diseased?' / 'denoise with median' / "
                        "'match features with ORB' / 'calibrate the camera' / 'show where the model looks'")
run = st.button("Run", type="primary")

if run:
    img = read(up)
    needs_image = mode != "calibrate" and not (mode.startswith("Auto") and route(request, True).intent == "calibrate")
    if img is None and needs_image:
        st.warning("Please upload an image first.")
        st.stop()
    if img is None:   # calibration without an input image: use the first chessboard sample
        img = cv2.imread("data/calib/left01.jpg")

    r = route(request, True, default=None if mode.startswith("Auto") else mode)
    if not mode.startswith("Auto"):
        r.intent, r.reason = mode, "mode chosen manually"
    st.info(f"**Detected task:** `{r.intent}` - {r.reason}. Scores: " +
            ", ".join(f"{k}={v}" for k, v in r.scores.items()))

    kwargs = dict(r.params)
    if r.intent == "features":
        kwargs["image2"] = read(up2)
    if r.intent == "calibrate" and calib_ups:
        import tempfile
        from pathlib import Path
        tmp = Path(tempfile.mkdtemp())
        for i, u in enumerate(calib_ups):
            (tmp / f"c{i}.jpg").write_bytes(u.getvalue())
        kwargs["calib_paths"] = sorted(tmp.glob("*.jpg"))
    try:
        with st.spinner(f"Running {r.intent} tool..."):
            result = TOOLS[r.intent](img, **kwargs)
    except Exception as e:   # surface tool errors to the user instead of a stack trace
        st.error(f"Tool `{r.intent}` failed: {e}")
        st.stop()

    st.subheader("Result")
    st.markdown(result.text.replace("\n", "  \n"))
    cols = st.columns(min(3, max(1, len(result.images))))
    for i, (title, im) in enumerate(result.images):
        with cols[i % len(cols)]:
            st.image(im, caption=title, width="stretch", clamp=True)
