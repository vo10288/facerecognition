<img width="1441" height="1028" alt="3" src="https://github.com/user-attachments/assets/7cbe5960-c360-4a9d-8691-88211e36ab8b" />


FaceCompare Multi-Face
README and Installation Guide
macOS • Windows • Linux
Setup with venv or Conda, offline InsightFace models, and build notes
Recommended baseline: Python 3.11.
For the application itself, keep InsightFace models outside the public repository and place them in insightface_local/models/buffalo_l/.
Quick start at a glance
Platform
Recommended environment
Best starting choice
macOS Intel / Apple Silicon
Conda or venv
Conda + Python 3.11 + pip install insightface
Windows 11
Conda or venv
Conda + Python 3.11; if build from source is needed, install Microsoft Build Tools
Linux
venv or Conda
venv + Python 3.11 or Conda + Python 3.11
Project files referenced in this guide
    • 1:1 image comparison: face_compare_multiface_scrolling_canvas_offline_fixed.py
    • Directory vs directory / video comparison: face_compare_arcface_mediapipe_forensic_DIR_multiface_canvas_offline_fixed_MacHome.py
    • Local model directory: insightface_local/models/buffalo_l/
1. Directory layout
Keep the Python files, build scripts, and the local model directory together inside the main project folder. A practical structure is the following:
project-root/
├─ face_compare_multiface_scrolling_canvas_offline_fixed.py
├─ face_compare_arcface_mediapipe_forensic_DIR_multiface_canvas_offline_fixed_MacHome.py
├─ compila-offline-1-1-....sh / .bat
├─ compila-offline-D-D-....sh / .bat
├─ icon.icns or icon.ico
└─ insightface_local/
   └─ models/
      └─ buffalo_l/
         ├─ 1k3d68.onnx
         ├─ 2d106det.onnx
         ├─ det_10g.onnx
         ├─ genderage.onnx
         └─ w600k_r50.onnx
The offline versions of the scripts are configured to look for the ONNX models locally instead of downloading them from GitHub at runtime.
2. Python package requirements
Use these base requirements for the runtime environment:
numpy==1.26.4
mediapipe==0.10.21
onnxruntime==1.23.2
pillow==12.1.1
simsimd
Install InsightFace separately with pip after the base requirements. This keeps the main requirements stable and mirrors the tested workflow used during development.
3. Installation on macOS (Intel and Apple Silicon)
The recommended baseline on macOS is Python 3.11. Both Conda and venv can be used.
3.1 Conda workflow
conda create -n face311 python=3.11 -y
conda activate face311
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install insightface
Verification:
python -c "import numpy, mediapipe, cv2, onnxruntime, insightface; print('numpy', numpy.__version__); print('mediapipe', mediapipe.__version__); print('cv2 ok'); print('onnxruntime', onnxruntime.__version__); print('insightface ok')"
3.2 venv workflow
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install insightface
    • For PyInstaller on macOS, use --add-data "insightface_local:insightface_local" because macOS uses : as the separator.
    • For the application icon, prefer .icns on macOS.
    • If PyInstaller complains about Xcode license or lipo, run sudo xcodebuild -license accept and verify Command Line Tools are installed.
4. Installation on Windows 11
Use Python 3.11. Conda is usually the easiest path, but venv works too.
4.1 Conda workflow
conda create -n py3.11FaceCompARC-MED python=3.11 -y
conda activate py3.11FaceCompARC-MED
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install insightface
If pip tries to build InsightFace from source instead of using a wheel, install Microsoft C++ Build Tools and run the install from a Developer Command Prompt or after initializing VsDevCmd.bat.
"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64
conda activate py3.11FaceCompARC-MED
pip install insightface==0.7.3
4.2 venv workflow
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install insightface
    • For PyInstaller on Windows, use --add-data "insightface_local;insightface_local" because Windows uses ; as the separator.
    • For the application icon, use a real .ico file.
    • If setup breaks because mediapipe requires numpy<2, reinstall numpy==1.26.4 and then reinstall the pinned requirements.
5. Installation on Linux
Linux generally works well with either venv or Conda. Python 3.11 remains the recommended baseline.
5.1 venv workflow
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install insightface
5.2 Conda workflow
conda create -n face311 python=3.11 -y
conda activate face311
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install insightface
If a Linux machine is very locked down, verify that the local ONNX model files are present before first launch so no network access is needed at runtime.
6. Running the applications
1:1 image comparison:
python face_compare_multiface_scrolling_canvas_offline_fixed.py
Directory vs directory / video comparison:
python face_compare_arcface_mediapipe_forensic_DIR_multiface_canvas_offline_fixed_MacHome.py
The directory version stores writable data under the user profile on macOS in:
~/Library/Application Support/FaceCompareMultiFace/
7. Building executables with PyInstaller
Start with onedir builds. They are easier to debug than onefile builds for projects that use OpenCV, InsightFace, ONNX Runtime, and Tkinter.
7.1 Windows example
pyinstaller ^
  --noconfirm ^
  --clean ^
  --windowed ^
  --onedir ^
  --name FaceCompareMultiFace ^
  --add-data "insightface_local;insightface_local" ^
  --collect-all insightface ^
  --collect-all onnxruntime ^
  --collect-all mediapipe ^
  --collect-submodules PIL ^
  --copy-metadata insightface ^
  --copy-metadata onnxruntime ^
  --icon="icon.ico" ^
  face_compare_multiface_scrolling_canvas_offline_fixed.py
7.2 macOS / Linux example
pyinstaller \
  --noconfirm \
  --clean \
  --windowed \
  --onedir \
  --name FaceCompareMultiFace \
  --add-data "insightface_local:insightface_local" \
  --collect-all insightface \
  --collect-all onnxruntime \
  --collect-all mediapipe \
  --collect-submodules PIL \
  --copy-metadata insightface \
  --copy-metadata onnxruntime \
  --icon="FaceIcon.icns" \
  face_compare_multiface_scrolling_canvas_offline_fixed.py
8. Troubleshooting
Issue
Practical fix
PyInstaller says pkg_resources is missing
Upgrade PyInstaller to a modern 6.x release and retry. Older 5.x builds can break with newer setuptools environments.
macOS build fails with Xcode / lipo license errors
Run sudo xcodebuild -license accept, then rebuild. If needed, install the Apple Command Line Tools with xcode-select --install.
InsightFace tries to reach GitHub on first launch
Check that insightface_local/models/buffalo_l contains all required ONNX files and that the offline Python file is the one being launched.
The app cannot create index or temp directories on macOS
Use the MacHome build, which stores writable files under ~/Library/Application Support/FaceCompareMultiFace/ instead of the current working directory.
mediapipe breaks after installing another package
Reinstall numpy==1.26.4 and then reinstall the pinned requirements. MediaPipe 0.10.21 requires NumPy below version 2.
9. Licensing and distribution note
The repository source code can be published under your chosen open-source license, but keep the InsightFace pretrained models outside the public repository and outside public executable bundles unless you have verified the model-specific redistribution terms. A clean approach is to publish source code, build scripts, and executables that expect the user to place the local model files manually.
