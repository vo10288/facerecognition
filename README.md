<img width="1441" height="1028" alt="3" src="https://github.com/user-attachments/assets/7cbe5960-c360-4a9d-8691-88211e36ab8b" />

# Face Compare MultiFace

Face comparison toolkit with:

- **1:1 image vs image** multi-face comparison
- **directory vs directory** indexing and comparison
- support for **images** and optional **video frame sampling**
- **MediaPipe** for yellow facial keypoints
- **ArcFace / InsightFace** for embeddings and biometric comparison
- responsive GUI previews
- HTML / CSV / package export
- support for **offline local InsightFace models**


## requirements.txt

Recommended `requirements.txt`:

```txt
numpy==1.26.4
mediapipe==0.10.21
onnxruntime==1.23.2
pillow==12.1.1
simsimd
```

Install `insightface` separately after the base requirements.

---

## Installation with venv

### Windows 11

Install Python 3.11 first, then:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install insightface
```

Verify:

```bat
python -c "import numpy, mediapipe, cv2, onnxruntime, insightface; print('numpy', numpy.__version__); print('mediapipe', mediapipe.__version__); print('cv2 ok'); print('onnxruntime', onnxruntime.__version__); print('insightface ok')"
```

### macOS

If Python 3.11 is already installed:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install insightface
```

Verify:

```bash
python -c "import numpy, mediapipe, cv2, onnxruntime, insightface; print('numpy', numpy.__version__); print('mediapipe', mediapipe.__version__); print('cv2 ok'); print('onnxruntime', onnxruntime.__version__); print('insightface ok')"
```

### Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install insightface
```

Verify:

```bash
python -c "import numpy, mediapipe, cv2, onnxruntime, insightface; print('numpy', numpy.__version__); print('mediapipe', mediapipe.__version__); print('cv2 ok'); print('onnxruntime', onnxruntime.__version__); print('insightface ok')"
```

---

## Installation with Conda

### Windows 11

```bat
conda create -n face311 python=3.11 -y
conda activate face311
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install insightface
```

### macOS Intel / Apple Silicon

```bash
conda create -n face311 python=3.11 -y
conda activate face311
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install insightface
```

### Linux

```bash
conda create -n face311 python=3.11 -y
conda activate face311
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install insightface
```


## Running the programs

### 1:1 image vs image multi-face

```bash
python face_compare_multiface_scrolling_canvas_offline_fixed.py
```

### Directory vs directory multi-face

```bash
python face_compare_arcface_mediapipe_forensic_DIR_multiface_canvas_offline_fixed.py
```

### macOS writable-home variant

```bash
python face_compare_arcface_mediapipe_forensic_DIR_multiface_canvas_offline_fixed_MacHome.py
```

This variant stores writable data under:

```text
~/Library/Application Support/FaceCompareMultiFace/
```

for example:

- indexes
- temp files
- other writable support files

---

## Functional overview

### 1:1 mode

The image-vs-image program:

- loads one image A
- loads one image B
- detects **all faces** in both images
- compares every face in A against every face in B
- sorts matches by score
- lets you scroll through results in the GUI

### Directory mode

The directory-vs-directory program:

- scans directory A
- scans directory B
- supports optional recursive traversal
- supports optional video processing
- samples frames according to configured frame step
- detects **all faces** in images and sampled video frames
- stores PKL indexes
- compares all indexed faces A vs B

---

## Comparison logic

The biometric match uses:

- **ArcFace / InsightFace** for face embeddings
- **cosine similarity** for the final score

MediaPipe is used for:

- facial landmarks
- yellow keypoint rendering
- visual support on the selected face

MediaPipe is **not** used for the final biometric score.

---

## Score interpretation

The similarity score is based on cosine similarity.

Theoretical score range:

- minimum theoretical value: `-1.00`
- maximum theoretical value: `+1.00`

GUI threshold range:

- minimum threshold: `0.00`
- maximum threshold: `1.00`

Typical default threshold:

- `0.55`

Interpretation used in the GUI:

- `score >= threshold` → high compatibility
- `score >= threshold - 0.20` → medium compatibility
- lower values → low compatibility

---

## Exports

The programs can export:

### HTML

Contains:

- ranked matches
- annotated previews
- file paths
- timestamps for video frames
- MD5 values
- technical comparison explanation

### CSV

Contains:

- rank
- ArcFace score
- verdict
- source paths
- frame numbers
- timestamps
- face indexes
- hashes

### Package export

Can contain:

- `report.html`
- `report.csv`
- cropped faces
- history entries
- support files

---

## Disclaimer

This software is a technical comparison tool.

Similarity scores alone do **not** prove identity.
All results should be reviewed by a qualified human operator.

purpose, legal admissibility, investigative sufficiency, or evidentiary reliability. Use of this software is entirely at the user's own risk.


This software is provided solely as a technical support and investigative tool and must be used responsibly, lawfully, and with due professional judgment.

Users are solely responsible for ensuring that any use of this software complies with all applicable laws, regulations, internal policies, and professional standards, including, but not limited to, data protection, privacy, and evidence-handling requirements. In jurisdictions subject to the GDPR and similar privacy regulations, any processing of personal data, biometric data, images, or videos must be carefully assessed for lawfulness, necessity, proportionality, data minimization, and purpose limitation before use.

This software may process image and video files that may contain personal data, sensitive data, biometric information, or material generated, modified, or synthetically produced by artificial intelligence. Users must therefore exercise particular care in assessing the authenticity, provenance, integrity, and evidentiary value of such material. Media generated, modified, or otherwise manipulated by AI can be misleading and should never be considered inherently reliable.

Similarity scores, facial comparisons, clustering results, or any other automatic output produced by this software do not constitute proof of identity, authenticity, or evidentiary veracity. They are merely investigative indicators and must always be verified through independent review, source validation, contextual analysis, and, where necessary, appropriate forensic methodologies.

Especially in criminal investigations, these tools can provide valuable investigative support, but reliance on primary sources remains essential. Particular attention must be paid to the original source of the data, chain of custody, metadata, acquisition method, reproducibility, and any possibility of tampering, synthetic generation, or alteration. Conclusions should never be based solely on automatic comparisons or derived output.

The authors and distributors of this software make no representations or warranties, express or implied, regarding fitness for a particular purpose, legal admissibility, detective sufficiency, or evidentiary reliability. Use of this software is entirely at the user's own risk.
