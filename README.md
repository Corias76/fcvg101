# Leaf Inspection Assistant (FCVG101)

Computer-vision MVP for an organic tomato farm: camera calibration, classical image processing, a CNN leaf-disease classifier (PlantVillage tomato, 10 classes) and a Streamlit assistant that routes plain-language requests to the right tool.

## Layout
- `cv_engine/` - Task 1: `calibration.py`, `filtering.py`, `features.py` (Harris from scratch, SIFT/ORB + ratio test + RANSAC)
- `dl/` - Task 2: `data_pipeline.py`, `model.py`, `train.py`, `evaluate.py`, `inference.py` (incl. Grad-CAM)
- `assistant/` + `app.py` - Task 3: intent router, tool wrappers, Streamlit UI
- `tests/` - routing tests; `run_task1.py` - Task 1 demo; `build_report.py` - builds `report/FCVG101_report.pdf`
- `models/tomato_cnn.pt` - trained model (best val macro-F1), `models/calibration.npz` - calibration result

## Requirements
Python 3.10 or newer. Dependencies are listed in `requirements.txt` (CPU-only PyTorch is enough: training takes about 14 min on 2 CPU cores).

## Run
```bash
pip install -r requirements.txt
git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset data/pv
(cd data/pv && git sparse-checkout set raw/color)
# put chessboard images (OpenCV samples/data/left*.jpg) in data/calib/
python run_task1.py          # Task 1 figures + outputs/task1_summary.json
python -m dl.data_pipeline   # clean, split, cache (~30 s)
python -m dl.train --epochs 14   # ~14 min on 2 CPU cores (add --arch mobilenet_v3_small --pretrained for transfer learning)
python -m dl.evaluate        # test metrics, confusion matrix, curves
streamlit run app.py         # assistant at http://localhost:8501
```
Test-set result: accuracy 0.9717, macro-F1 0.9679 (2,717 images). See the report for limitations (PlantVillage is lab-condition data, so field accuracy will be lower).

## Quick check without retraining
The trained model and calibration are already in `models/` and the chessboard images are in `data/calib/`, so after `pip install -r requirements.txt` you can run `streamlit run app.py` straight away (upload any tomato leaf photo) and `python -m tests.test_router` for the routing test. Training and evaluation need the PlantVillage download above.

