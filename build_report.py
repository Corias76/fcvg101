"""Builds report/FCVG101_report.pdf (single PDF: text, figures, numerical results, selectable source code)."""
import io
import json
import textwrap
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle)

OUT = Path("outputs")
T1 = json.loads((OUT / "task1_summary.json").read_text())
T2 = json.loads((OUT / "task2_metrics.json").read_text())
CL = json.loads(Path("data/cache/cleaning_report.json").read_text())
HIST = json.loads((OUT / "training_history.json").read_text())

ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.6, leading=13, alignment=TA_JUSTIFY, spaceAfter=5)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#1f3a5f"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1f3a5f"))
CAP = ParagraphStyle("CAP", parent=B, fontSize=8, leading=10, textColor=colors.HexColor("#444444"), alignment=0)
CODE = ParagraphStyle("CODE", fontName="Courier", fontSize=5.9, leading=7.1)
story = []


def P(t, st=B):
    story.append(Paragraph(t, st))


def bullets(items):
    for t in items:
        story.append(Paragraph("•&nbsp;&nbsp;" + t, ParagraphStyle("bl", parent=B, leftIndent=14, firstLineIndent=-10, spaceAfter=3)))


def fig(name, caption, width=16.5 * cm, max_px=1500, q=78):
    im = PILImage.open(OUT / name).convert("RGB")
    if im.width > max_px:
        im = im.resize((max_px, int(im.height * max_px / im.width)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True)
    buf.seek(0)
    story.append(Image(buf, width=width, height=width * im.height / im.width))
    story.append(Paragraph(caption, CAP))
    story.append(Spacer(1, 5))


def table(rows, widths=None, header=True, fs=8):
    cell = ParagraphStyle("cell", parent=B, fontSize=fs, leading=fs + 2.2, alignment=0, spaceAfter=0)
    cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
    rows = [[Paragraph(str(c), cellb if (header and i == 0) else cell) for c in r] for i, r in enumerate(rows)]  # wrap long cells
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [("FONTSIZE", (0, 0), (-1, -1), fs), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe7f2")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 6))


# =============================================================================== title
P("FCVG101 Introduction to Computer Vision - Coursework Report", ParagraphStyle("T", parent=H1, fontSize=19, spaceAfter=2))
P("<b>Leaf Inspection Assistant:</b> a computer-vision MVP for an organic tomato farm", ParagraphStyle("S", parent=B, fontSize=11.5))
P("Student name / ID: <b>Nguyen Tien Dung / [student ID]</b> &nbsp;|&nbsp; Code repository: "
  "<b>https://github.com/Corias76/fcvg101</b> &nbsp;|&nbsp; Trained model (shared link): "
  "<b>https://github.com/Corias76/fcvg101/blob/main/models/tomato_cnn.pt</b>", B)

H = "<b>Summary.</b> "
P(H + "The system plays the role of an inspection assistant for an organic tomato farm. It (a) models the camera with the pinhole "
  "model and calibrates it from a chessboard, (b) processes images with classical filtering, gradient, frequency-domain and "
  "feature-matching techniques, and (c) diagnoses leaf disease with a CNN trained from scratch on 10 PlantVillage tomato classes. "
  f"On a held-out test set of {T2['n_test']} images the CNN reaches <b>{T2['test_accuracy']:.1%} accuracy</b> and "
  f"<b>{T2['test_macro_f1']:.3f} macro-F1</b>. A Streamlit app reads a free-text request, identifies the task, runs the right module "
  "and shows the visual result. Every design choice below is followed by the reason it was made; Section 6 lists the limitations "
  "that a reader should keep in mind when interpreting the numbers.")

P("1. Scenario, dataset and justification", H1)
P("<b>Scenario.</b> An organic farm has no routine lab access, so a farm assistant that tells a grower whether a leaf is healthy, and "
  "which disease it shows, from a phone or camera photo is a realistic MVP. Organic growers cannot rely on preventive "
  "chemical spraying, so early and specific diagnosis has direct value.")
P("<b>Dataset: PlantVillage (tomato subset, 10 classes).</b> Chosen because (i) it is public and widely used, so results can be "
  "compared with the literature; (ii) tomato has the most disease classes of any single crop in the dataset (9 diseases + healthy), "
  "which makes the classification task non-trivial; (iii) the images are 256x256 colour leaf photographs, well suited to a small CNN. "
  "The subset (rather than all 38 classes across 14 crops) keeps training feasible on the 2-core CPU used for development and keeps the "
  "scenario coherent: one crop, one farm. The same code runs on all classes by setting <font face='Courier'>CROP=None</font> in "
  "<font face='Courier'>dl/data_pipeline.py</font>.")
P("<b>Requirement (a) calibration data.</b> OpenCV's public chessboard sample images (13 usable views, 640x480, 9x6 inner corners) are used, because no "
  "photographs from the target farm camera exist. The pipeline is unchanged for the user's own camera: put 10-20 chessboard "
  "photos in <font face='Courier'>data/calib/</font>.")

# =============================================================================== Task 1
P("2. Task 1 - Core image-analysis engine", H1)
P("<b>Toolkit choice.</b> OpenCV + NumPy/SciPy are used for the standard components because they are fast, tested and the industry "
  "norm. Three techniques are written from scratch to show understanding: the explicit pinhole projection, a 2-D correlation "
  "routine, and the Harris corner detector. Each from-scratch routine is checked numerically against the OpenCV equivalent, which is "
  "the reason to implement them at all: a check of agreement turns a re-implementation into evidence of correctness. The code is split "
  "into <font face='Courier'>cv_engine/calibration.py</font>, <font face='Courier'>filtering.py</font> and <font face='Courier'>features.py</font>, "
  "each exposing small pure functions with docstrings, so that the app (Task 3) can call them without any change.")

P("2.1 Image formation and camera calibration", H2)
P("The pinhole model maps a world point X to pixel x ~ K[R|t]X with K = [[fx,0,cx],[0,fy,cy],[0,0,1]]. Real lenses add radial and tangential "
  "distortion (k1,k2,p1,p2,k3). Calibration finds K and the distortion from several views of a known planar pattern: with a plane, each view "
  "gives a homography between board and image, and several homographies over-constrain K (Zhang's method, implemented by "
  "<font face='Courier'>cv2.calibrateCamera</font>). A chessboard is used because its corners can be located to sub-pixel accuracy; "
  "<font face='Courier'>cornerSubPix</font> is applied since corner-localisation noise limits calibration accuracy.")
K = T1["calibration"]["K"]
d = T1["calibration"]["dist"]
table([["Quantity", "Result"],
       ["Views used", f"{T1['calibration']['n_images']} of 13"],
       ["fx, fy (px)", f"{K[0][0]:.1f}, {K[1][1]:.1f}"],
       ["Principal point (cx, cy) px", f"{K[0][2]:.1f}, {K[1][2]:.1f}"],
       ["Distortion k1, k2, p1, p2, k3", ", ".join(f"{v:.4f}" for v in d)],
       ["RMS reprojection error", f"{T1['calibration']['rms']:.3f} px"],
       ["max |explicit pinhole - cv2.projectPoints|", f"{T1['calibration']['pinhole_vs_cv2_px']:.1e} px"]], [7 * cm, 8 * cm])
P("<b>Reading the result.</b> fx and fy are almost equal (536.1 vs 536.0), as expected for a sensor with square pixels, which is a sanity check "
  "on the estimate. The principal point lies near the image centre (320,240), also expected. The RMS error of 0.41 px is below the 1 px "
  "level that is usually accepted as a good calibration. The negative k1 (-0.265) indicates barrel distortion, visible in the undistorted image "
  "where the curved board edges become straight. The explicit pinhole projection I wrote agrees with OpenCV to 3e-5 px, which confirms that the model "
  "is implemented correctly. The unit of the board square is arbitrary (set to 1), so translations are in square units; K and the distortion are in "
  "pixels and unaffected.")
fig("t1_calibration.png", "Figure 1. Left: detected corners (sub-pixel). Middle: original. Right: undistorted with alpha=1 (all pixels kept, hence the curved black borders).", 16.5 * cm)

P("2.2 Filtering, gradients and frequency analysis", H2)
P("<b>Denoising.</b> Gaussian (sigma=25) noise and salt-and-pepper (6%) noise were added to a grayscale leaf and removed with 5x5 box, Gaussian, median and bilateral filters. "
  "PSNR against the clean image measures quality. Two noise types are used because the best filter depends on the noise: linear filters "
  "average impulses into their neighbours, whereas the median ignores outliers.")
g, sp = T1["filtering"]["psnr_gaussian_noise"], T1["filtering"]["psnr_salt_pepper"]
table([["PSNR (dB)", "noisy", "box", "gaussian", "median", "bilateral"],
       ["Gaussian noise"] + [f"{g[k]:.2f}" for k in ("noisy", "box 5x5", "gaussian 5x5", "median 5x5", "bilateral")],
       ["Salt & pepper"] + [f"{sp[k]:.2f}" for k in ("noisy", "box 5x5", "gaussian 5x5", "median 5x5", "bilateral")]],
      [3.6 * cm] + [2.4 * cm] * 5)
P("<b>Reading the result.</b> For Gaussian noise the Gaussian and bilateral filters win (28.1 and 28.5 dB), matching theory: a Gaussian kernel is the "
  "linear filter that best averages independent zero-mean noise while weighting near pixels more, and the bilateral filter additionally preserves edges. "
  "For salt-and-pepper noise the median wins clearly (29.2 dB vs 27.1 for box), because outliers are removed rather than spread. The bilateral filter is poor on impulse noise "
  "(21.0 dB) because an extreme pixel is very different from its neighbours, so the range weighting gives it almost no help from them. "
  "The app therefore defaults to the median filter for speckle.")
fig("t1_denoise.png", "Figure 2. Top row: Gaussian noise, then box / Gaussian / median. Bottom row: salt-and-pepper noise, then box / Gaussian / median.", 16.5 * cm)
P("<b>Gradients and edges.</b> The image is lightly Gaussian-smoothed before differentiating, because derivatives amplify noise. Sobel gives gradient magnitude and direction. "
  "Canny adds non-maximum suppression and hysteresis thresholding to give thin, connected edges. The thresholds are set automatically from the image's own gradients "
  "(high = 90th percentile of Sobel magnitude, low = 0.4 x high). This replaced a median-intensity rule from common tutorials, which produced an almost empty edge map on "
  "low-contrast leaf images in testing: that rule links thresholds to brightness rather than to contrast.")
fig("t1_edges.png", "Figure 3. Grayscale, Sobel magnitude and Canny edges.", 15 * cm)
P("<b>Frequency domain.</b> The 2-D FFT shows which spatial frequencies carry the energy. A Gaussian transfer function is used for low/high-pass filtering instead of an ideal "
  "(brick-wall) cut-off, since an ideal mask has a sinc impulse response and causes ringing. Low-pass keeps the smooth leaf shape and illumination; high-pass keeps veins and lesion "
  "borders, which is the information a disease detector needs. The from-scratch correlation routine matches <font face='Courier'>cv2.filter2D</font> to "
  f"{T1['filtering']['naive_conv_max_diff']:.0e} grey levels.")
fig("t1_fft.png", "Figure 4. Grayscale, log-magnitude spectrum, Gaussian low-pass and high-pass (sigma = 25 frequency bins).", 16.5 * cm)

P("2.3 Feature detection and matching", H2)
P("<b>Harris from scratch.</b> The structure tensor M (Gaussian-weighted sums of Ix^2, IxIy, Iy^2) has two large eigenvalues at a corner. R = det(M) - k trace(M)^2 avoids computing eigenvalues, "
  "with k = 0.05 inside the usual 0.04-0.06 range. Corners are the local maxima of R (non-maximum suppression, 7x7) above 1% of the maximum response.")
P("<b>SIFT and ORB with the ratio test.</b> SIFT (float, L2 distance) is scale- and rotation-invariant and accurate; ORB (binary, Hamming distance) is much faster, which matters on a phone. "
  "Matches use k-NN with k=2 and Lowe's ratio test: a match is kept only if the best distance is below 0.75 x the second-best, which removes ambiguous matches such as repeated vein textures. "
  "RANSAC then estimates a homography and separates geometrically consistent inliers.")
P("<b>How matching is evaluated.</b> A second 'view' is generated by warping the image with a known homography (rotation 12 deg, scale 0.9, mild perspective). Knowing the true homography lets me "
  "measure accuracy quantitatively: the homography error is the mean pixel distance between points mapped by the estimated and the true homography over a 50x50 grid.")
s, o = T1["features"]["sift"], T1["features"]["orb"]
table([["Method", "keypoints (view1/view2)", "ratio-test matches", "RANSAC inliers", "inlier rate", "homography error (px)"],
       ["SIFT", f"{s['kp1']} / {s['kp2']}", s["ratio_matches"], s["ransac_inliers"], f"{s['inlier_rate']:.1%}", f"{s['homography_error_px']:.2f}"],
       ["ORB", f"{o['kp1']} / {o['kp2']}", o["ratio_matches"], o["ransac_inliers"], f"{o['inlier_rate']:.1%}", f"{o['homography_error_px']:.2f}"]],
      [2 * cm, 3.6 * cm, 3 * cm, 2.6 * cm, 2.2 * cm, 3.4 * cm])
sw = T1["features"]["sift_ratio_sweep"]
table([["ratio threshold"] + list(sw.keys()), ["matches / inliers"] + [f"{v[0]} / {v[1]}" for v in sw.values()]], [3 * cm] + [2.1 * cm] * len(sw))
P("<b>Reading the result.</b> SIFT gives more accurate geometry (0.09 px vs 0.30 px error) and a 99% inlier rate, ORB returns more matches but with more outliers. The ratio sweep justifies the 0.75 threshold: "
  "going from 0.75 to 1.0 (no filtering) doubles the matches (493 to 1018) but inliers rise by only 7 (490 to 497), so almost all the extra matches are wrong. Thresholds of 0.6-0.8 give "
  "almost the same inliers, so the choice is not sensitive. The Harris detector found 184 well-spread corners; the raw count is not comparable with OpenCV's response map (which is thresholded per pixel, before non-maximum suppression), so I do not claim one is better.")
fig("t1_matches_sift.jpg", "Figure 5. SIFT inlier matches between the leaf and its warped second view.", 16 * cm)
fig("t1_harris.png", "Figure 6. Harris corners from the from-scratch implementation.", 8 * cm)

# =============================================================================== Task 2
P("3. Task 2 - Data pipeline and deep-learning model", H1)
P("3.1 Data pipeline (dl/data_pipeline.py)", H2)
P("The pipeline is five documented functions, run once by <font face='Courier'>python -m dl.data_pipeline</font>, which caches the result so that experiments are reproducible and fast.")
table([["Step", "What is done", "Why"],
       ["1 Ingestion", f"Scan class folders (folder = label): {CL['n_raw']} tomato images, 10 classes.", "PlantVillage stores one class per folder; labels come from the folder, so there is nothing to merge."],
       ["2 Cleaning", f"Unreadable: {CL['unreadable']}. Byte-identical duplicates (MD5): {CL['exact_duplicates']}. Visual duplicates within a class (256-bit dHash): {CL['near_duplicates']}. "
        f"Blurred: {CL['blurred']} (Laplacian variance < {CL['blur_threshold_laplacian_var']:.1f}). Degenerate (mean <15 or >240, or std <8): {CL['degenerate']}. Kept: {CL['n_clean']}.",
        "Duplicates would sit in both train and test and inflate accuracy. Blur threshold = median - 3.5 robust SD of log(Laplacian variance), so it adapts to the data instead of being a magic number; log because variance is heavy-tailed."],
       ["3 Split", f"Stratified 70/15/15: train {CL['n_train']}, val {CL['n_val']}, test {CL['n_test']}, seed 42.", "Stratification keeps the class proportions equal in all three sets, which matters for the 373-image mosaic-virus class. Test set is touched once, at the end."],
       ["4 Transformation", "Resize to 128x128 (INTER_AREA), RGB, scale to [0,1], standardise per channel with mean " + str(CL["mean"]) + " and std " + str(CL["std"]) + " computed on the training set only.",
        "128 px keeps lesion detail while making CPU training possible; INTER_AREA avoids aliasing when down-sampling. Statistics from train only, to avoid leakage."],
       ["5 Augmentation (train only)", "Random flips (h and v), rotation +/-30 deg, scale 0.85-1.15, brightness/contrast jitter, Gaussian noise (sd 0.05), applied to each batch with one affine grid-sample.",
        "A leaf has no canonical orientation, so flips/rotations keep the label. Colour jitter is mild because colour is diagnostic. Noise adds robustness. Batch-level tensor ops are much faster on CPU than per-image PIL transforms."]],
      [2.4 * cm, 7.3 * cm, 7.3 * cm], fs=7.3)
cd = CL["class_counts"]
P(f"<b>Class imbalance.</b> The largest class (Yellow Leaf Curl Virus, {cd['Tomato_Yellow_Leaf_Curl_Virus']}) is 14x larger than the smallest (mosaic virus, {cd['Tomato_mosaic_virus']}). "
  "Instead of discarding data (undersampling) or duplicating images (oversampling), the loss is weighted by (N/(K n_c))^0.5. The square root tempers full inverse-frequency "
  "weighting, which makes the rarest class dominate gradients. Macro-F1 is reported next to accuracy, since accuracy alone would hide poor performance on small classes.")
fig("t2_classdist.png", "Figure 7. Class distribution after cleaning.", 11 * cm)
fig("t2_augment.png", "Figure 8. One training image and four random augmentations.", 16.5 * cm)

P("3.2 Model (dl/model.py, dl/train.py)", H2)
P("<b>Architecture choice.</b> The task is image classification, so a CNN is appropriate: convolutions share weights across positions, so they detect a lesion wherever it is on the leaf, with far fewer parameters than a "
  "fully connected net. A compact VGG-style stack (Conv-BatchNorm-ReLU, with max-pooling) is used. A stride-2 stem and a second stride-2 convolution downsample early, since the cost of a convolution scales with feature-map area. "
  "BatchNorm stabilises and speeds training; global average pooling replaces large dense layers, cutting parameters and giving a spatial map for Grad-CAM; dropout (0.3) regularises the head. "
  "The model has 468,890 parameters. Transfer learning from an ImageNet MobileNetV3 is implemented as an option (<font face='Courier'>--arch mobilenet_v3_small --pretrained</font>); it was not used for the reported "
  "results because the development environment could not download pretrained weights, so the model is trained from scratch.")
table([["Hyper-parameter", "Value", "Reason"],
       ["Optimiser", "AdamW, weight decay 1e-4", "Adaptive steps converge quickly on small nets; decoupled decay regularises."],
       ["LR schedule", "OneCycle, max 3e-3", "Warm-up then annealing gave fast, stable convergence in a fixed 14-epoch budget."],
       ["Batch / epochs", "64 / 14", "~58 s per epoch on 2 CPU cores; validation F1 still rising at epoch 14 (see curves)."],
       ["Loss", "Cross-entropy, label smoothing 0.05, class weights (power 0.5)", "Smoothing limits over-confidence; weights handle imbalance."],
       ["Model selection", "Best validation macro-F1 checkpoint", "Macro-F1 treats all classes equally; validation (not test) data selects the model."],
       ["Seed", "42", "Reproducibility."]], [3.3 * cm, 5.4 * cm, 8.3 * cm], fs=7.6)

P("3.3 Evaluation", H2)
pc = T2["per_class"]
rows = [["Class", "Precision", "Recall", "F1", "Support"]]
for k, v in pc.items():
    if isinstance(v, dict) and k not in ("macro avg", "weighted avg"):
        rows.append([k, f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['f1-score']:.3f}", int(v["support"])])
rows.append(["<b>macro avg</b>".replace("<b>", "").replace("</b>", ""), f"{pc['macro avg']['precision']:.3f}", f"{pc['macro avg']['recall']:.3f}", f"{pc['macro avg']['f1-score']:.3f}", T2["n_test"]])
table(rows, [4.6 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm])
rob = T2["noise_robustness_acc_first1024"]
P(f"<b>Results.</b> Test accuracy {T2['test_accuracy']:.4f}, macro-F1 {T2['test_macro_f1']:.4f}, weighted-F1 {T2['test_weighted_f1']:.4f} (validation macro-F1 at the chosen epoch: {HIST[-1]['val_macro_f1']:.4f}; "
  "test and validation agree, so there is no sign that checkpoint selection over-fitted the validation set). The weakest classes are Target Spot (recall 0.896) and Early blight (precision 0.913); "
  "the confusion matrix shows why: 14 of the 22 Target Spot errors are predicted as healthy (which accounts for most of healthy's lower precision, 0.919, as healthy itself has perfect recall) and 6 as spider mites, "
  "probably because early Target Spot lesions are small; and Early blight's precision is lowered by 7 Late blight and 7 Septoria leaf spot leaves predicted as Early blight (all three show brown spotting). "
  "Training accuracy (0.976, with augmentation on) is close to validation accuracy (0.975), so the model is not over-fitting; the loss curves are still falling, suggesting more epochs would help slightly. "
  f"<b>Robustness:</b> adding Gaussian noise of sd 0.1 / 0.2 / 0.4 (normalised units) to the first 1024 test images gives accuracy {rob['0.1']:.1%} / {rob['0.2']:.1%} / {rob['0.4']:.1%} versus {rob['0.0']:.1%} clean: "
  "tolerant of moderate noise (the effect of the noise augmentation), but not of heavy noise, where the Task 1 median/Gaussian denoisers should be applied first.")
fig("t2_curves.png", "Figure 9. Training and validation loss/accuracy/macro-F1 per epoch.", 15.5 * cm)
fig("t2_confusion.png", "Figure 10. Test confusion matrix.", 11.5 * cm)
P("<b>Explainability (Grad-CAM).</b> Grad-CAM weights the last convolutional maps by the mean gradient of the predicted class score and shows the regions that supported the decision. It was added because a "
  "farmer needs to see why the system says 'disease'. Figure 11 shows an important caveat: for some images the heat is on lesions and leaf edges (late blight, mosaic virus), but for others it falls "
  "on cast shadows or background (the healthy example, the mold example's lower corner). This suggests that the model partly uses dataset-specific cues such as shadows, which is a known weakness of PlantVillage. See Section 6.")
fig("t2_gradcam.png", "Figure 11. Grad-CAM on four test images (top: true label; bottom: prediction and heat-map).", 16 * cm)
P("<b>Deployment artifact.</b> The best checkpoint is saved as <font face='Courier'>models/tomato_cnn.pt</font> (1.9 MB) together with class names, normalisation statistics and input size, so that the app applies exactly "
  "the training-time pre-processing (a common source of silent accuracy loss).")

# =============================================================================== Task 3
P("4. Task 3 - Assistant feature (application integration / tool use)", H1)
P("<b>Design.</b> A Streamlit app (<font face='Courier'>app.py</font>) lets the user upload an image (and optionally a second view or chessboard images), type a request in plain language, or force a mode. "
  "Three layers keep the code modular: <font face='Courier'>assistant/router.py</font> (intent recognition), <font face='Courier'>assistant/tools.py</font> (five tools with one uniform signature "
  "<font face='Courier'>tool(image, **options) -> ToolResult</font>) and the Task 1/Task 2 modules. The uniform signature is the 'tool use' interface: adding a tool means adding one function and one dictionary entry.")
table([["Intent", "Example request", "Module executed", "Output shown"],
       ["calibrate", "calibrate the camera and undistort", "cv_engine.calibration", "K, distortion, RMS error, undistorted image, detected corners"],
       ["filter", "denoise with median and show edges", "cv_engine.filtering", "noisy/denoised, PSNR, Sobel, Canny, FFT, low/high-pass"],
       ["features", "match features with ORB and Harris", "cv_engine.features", "inlier matches, counts, homography error, Harris corners"],
       ["classify", "is this leaf diseased?", "dl.inference", "top-3 classes with confidence, low-confidence warning"],
       ["explain", "show where the model looks", "dl.inference (+Grad-CAM)", "prediction plus Grad-CAM overlay"]], [2 * cm, 5 * cm, 4 * cm, 6 * cm], fs=7.6)
P("<b>Intent recognition - why keyword scoring.</b> With five closed intents, a transparent weighted-keyword scorer is deterministic, runs offline, shows its scores to the user (so a wrong routing can be seen and overridden in the Mode menu), "
  "and can be unit-tested, which an LLM call cannot guarantee. Each intent has stems with weights; the highest total wins; if nothing matches, the mode chosen by the user (or 'classify' if only an image is given) is used. "
  "A compound request that mentions both diagnosis and explanation runs the 'explain' tool, which includes the classifier, so that nothing the user asked for is dropped. "
  "Option extraction ('median', 'ORB', 'high-pass') is done with the same text.")
P("<b>Routing accuracy - honestly reported.</b> Development set (30 requests, used to write the vocabulary): 30/30. A first fresh set of 15 requests initially scored 10/15; "
  "the misses (e.g. 'remove the salt and pepper speckles', 'what regions made you say that') exposed missing vocabulary, which I added; that set is therefore no longer independent (15/15 after the fix). "
  "A second fresh set of 15 requests, written after the fix, scored <b>14/15</b>; the only miss was a request with no keyword ('link up the same points across the two pictures'), which fell back to the default. "
  "A rule-based router has this weakness by design; the Mode menu is the remedy and a small text classifier or an LLM would be the next step.")
fig("app_classify.png", "Figure 12. App: the request 'Is this leaf diseased? Show where the model looks' is routed to a tool and the result is displayed (screenshot cropped to the top).", 14 * cm, max_px=1100)
fig("app_filter.png", "Figure 13. App: 'denoise with median and show edges' routed to the filter tool (full page).", 11 * cm, max_px=900)
P("<b>Testing.</b> A headless Streamlit test (<font face='Courier'>streamlit.testing</font>) ran the calibration request without exceptions; all five tools were run through the router on a real leaf; "
  "tool errors are caught and shown to the user as a message rather than a stack trace.")

# =============================================================================== running
P("5. Instructions for running the whole system", H1)
for t in ["<font face='Courier'>pip install opencv-python numpy scipy scikit-image scikit-learn pandas matplotlib torch torchvision streamlit pillow</font>",
          "Download the PlantVillage dataset: <font face='Courier'>git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset data/pv</font>, then "
          "<font face='Courier'>cd data/pv && git sparse-checkout set raw/color</font>. Put chessboard images (OpenCV's <font face='Courier'>samples/data/left*.jpg</font>) in <font face='Courier'>data/calib/</font>.",
          "<font face='Courier'>python run_task1.py</font> - runs Task 1, writes figures and <font face='Courier'>outputs/task1_summary.json</font>.",
          "<font face='Courier'>python -m dl.data_pipeline</font> - cleans, splits and caches the data (about 30 s).",
          "<font face='Courier'>python -m dl.train --epochs 14</font> - trains (about 14 min on 2 CPU cores); <font face='Courier'>python -m dl.evaluate</font> - test metrics and figures.",
          "<font face='Courier'>streamlit run app.py</font> - starts the assistant at http://localhost:8501.",
          "<font face='Courier'>python -m tests.test_router</font> (also <font face='Courier'>test_router_heldout</font>, <font face='Courier'>test_router_heldout2</font>) - routing tests."]:
    bullets([t])

# =============================================================================== limitations
P("6. Limitations and honest caveats", H1)
bullets([
    "<b>Test accuracy is probably optimistic for real farms.</b> PlantVillage photographs are taken in a lab with plain backgrounds, and several photos can show the same leaf. A random split can therefore put near-identical "
    "leaves in train and test. Exact and visual duplicates were removed, but photos of the same leaf from different angles cannot be detected without leaf IDs, which the dataset does not provide. Field accuracy will be lower; I did not measure it.",
    "<b>Possible shortcut learning.</b> Grad-CAM (Figure 11) sometimes highlights shadows and background rather than lesions. A fair next step is testing on field images, or segmenting the leaf before classification.",
    "<b>Trained from scratch.</b> Pretrained weights could not be downloaded in the development environment. Transfer learning is implemented behind a flag and would likely raise accuracy and reduce training time.",
    "<b>Feature matching was evaluated on a synthetic second view</b> (a known homography), which is easier than a real change of viewpoint with depth. The numbers show correctness of the implementation, not performance in the wild.",
    "<b>Calibration used public sample images</b>, not the farm camera; the K reported is for that camera and must be re-estimated for another one.",
    "<b>Scope.</b> Ten tomato classes only; depth/motion measurement (optional in the brief) is not implemented; the router is rule-based.",
])

# =============================================================================== code appendix
story.append(PageBreak())
P("Appendix - Source code (selectable text)", H1)
files = ["cv_engine/calibration.py", "cv_engine/filtering.py", "cv_engine/features.py", "run_task1.py",
         "dl/data_pipeline.py", "dl/model.py", "dl/train.py", "dl/evaluate.py", "dl/inference.py", "make_report_figures.py",
         "assistant/router.py", "assistant/tools.py", "app.py", "tests/test_router.py"]
for f in files:
    P(f"<font face='Courier'>{f}</font>", H2)
    code = "\n".join("\n".join(textwrap.wrap(l, 150, subsequent_indent="        ", break_long_words=True, replace_whitespace=False) or [""])
                     for l in Path(f).read_text().expandtabs(4).splitlines())
    story.append(Preformatted(code, CODE))


def footer(c, d):
    c.saveState()
    c.setFont("Helvetica", 7.5)
    c.drawString(2 * cm, 1.1 * cm, "FCVG101 - Leaf Inspection Assistant")
    c.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {d.page}")
    c.restoreState()


Path("report").mkdir(exist_ok=True)
SimpleDocTemplate("report/FCVG101_report.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                  bottomMargin=1.8 * cm, title="FCVG101 Coursework Report").build(story, onFirstPage=footer, onLaterPages=footer)
print("built report/FCVG101_report.pdf")
