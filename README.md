<img width="1441" height="1028" alt="3" src="https://github.com/user-attachments/assets/7cbe5960-c360-4a9d-8691-88211e36ab8b" />


# Face Compare Directories v3 - MediaPipe + ArcFace

Face comparison software for **Directory A vs Directory B** with support for:

- recursive directory scanning
- image files
- optional video processing
- face extraction from sampled video frames
- MediaPipe yellow keypoints
- ArcFace / InsightFace embeddings
- massive A vs B comparison
- HTML export
- CSV export
- forensic package export
- PKL index caching per directory

---

## Recommended Environment

This project is currently configured for:

- **macOS Intel**
- **Conda**
- **Python 3.11**

This is the recommended setup because `insightface` is more reliable on **conda-forge** for **macOS-64** than via a direct `pip` build from source.

---

## Create the Conda environment

```bash
conda create -n face311 python=3.11 -y
conda activate face311

 ANCHE NO !conda install -c conda-forge insightface -y

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

numpy==1.26.4
mediapipe==0.10.21
onnxruntime==1.23.2
pillow==12.1.1
simsimd
insightface


---

## Nota pratica

**non inserire `insightface` nel `requirements.txt`**.  
Per macOS Intel è meglio continuare così:

```bash
conda install -c conda-forge insightface -y
pip install -r requirements.txtn
