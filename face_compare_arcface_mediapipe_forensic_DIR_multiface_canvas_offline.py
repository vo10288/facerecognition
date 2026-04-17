#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import base64
import csv
import hashlib
import os
import pickle
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

try:
    from insightface.app import FaceAnalysis
except Exception as exc:
    FaceAnalysis = None
    INSIGHTFACE_ERROR = exc
else:
    INSIGHTFACE_ERROR = None


APP_TITLE = "Face Compare Directories v4 - By Visi@n antonio@broi.it"
DEFAULT_THRESHOLD = 0.55
EXPORT_DIRNAME = "face_compare_exports_v4"
HISTORY_CSV = "face_compare_history_v4.csv"
INDEX_DIRNAME = "face_compare_indexes"

WINDOW_BG = "#111111"
PANEL_BG = "#1b1b1b"
TEXT_FG = "#f3f3f3"
MUTED = "#bbbbbb"
ACCENT = "#ffd400"
GOOD = "#1db954"
MID = "#f4c542"
BAD = "#e74c3c"

PREVIEW_SIZE = (640, 480)
CROP_SIZE = (300, 300)
YELLOW_BGR = (0, 255, 255)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpeg", ".mpg", ".wmv"}


@dataclass

INSIGHTFACE_LOCAL_DIRNAME = "insightface_local"
INSIGHTFACE_MODEL_NAME = "buffalo_l"
INSIGHTFACE_REQUIRED_FILES = [
    "1k3d68.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "genderage.onnx",
    "w600k_r50.onnx",
]


def get_runtime_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_insightface_root() -> Path:
    return get_runtime_base_dir() / INSIGHTFACE_LOCAL_DIRNAME


def get_insightface_model_dir() -> Path:
    return get_insightface_root() / "models" / INSIGHTFACE_MODEL_NAME


def validate_local_insightface_models() -> Path:
    model_dir = get_insightface_model_dir()
    missing = [name for name in INSIGHTFACE_REQUIRED_FILES if not (model_dir / name).exists()]
    if missing:
        raise RuntimeError(
            "Modelli InsightFace locali mancanti.\n"
            f"Cartella attesa: {model_dir}\n"
            "Struttura richiesta:\n"
            f"{get_insightface_root()}\\models\\{INSIGHTFACE_MODEL_NAME}\\\n"
            "File mancanti: " + ", ".join(missing) + "\n\n"
            "Copia i modelli ONNX locali in questa cartella per evitare il download da GitHub."
        )
    return model_dir

class FaceResult:
    source_path: str
    source_type: str
    frame_index: int
    timestamp_sec: float
    face_index: int
    total_faces_in_source: int
    display_name: str
    original_bgr: np.ndarray
    annotated_bgr: np.ndarray
    crop_bgr: Optional[np.ndarray]
    embedding: Optional[np.ndarray]
    similarity_ready: bool
    face_count_arcface: int
    mp_landmark_count: int
    bbox: Optional[Tuple[int, int, int, int]]
    warning: str
    md5: str
    sha256: str


@dataclass
class MatchRecord:
    score: float
    verdict: str
    source_a: str
    source_b: str
    display_a: str
    display_b: str
    type_a: str
    type_b: str
    frame_a: int
    frame_b: int
    time_a: float
    time_b: float
    face_idx_a: int
    face_idx_b: int
    total_faces_a: int
    total_faces_b: int
    md5_a: str
    md5_b: str


def sanitize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\-\.]+", "_", name, flags=re.UNICODE)
    return name or "index"


def make_index_path(base_dir: str) -> Path:
    base = Path.cwd() / INDEX_DIRNAME
    base.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_name(Path(base_dir).name)
    return base / f"{safe_name}.pkl"


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def area_from_bbox(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def join_notes(a: str, b: str) -> str:
    if a and b:
        return a + " | " + b
    return a or b


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = v1.astype(np.float32)
    v2 = v2.astype(np.float32)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def append_csv_row(csv_path: Path, header: list[str], row: list[str]) -> None:
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


def img_to_b64(bgr: Optional[np.ndarray]) -> str:
    if bgr is None:
        blank = np.zeros((220, 220, 3), dtype=np.uint8)
        cv2.putText(blank, "N/D", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        bgr = blank
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Impossibile convertire immagine in PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def escape_html(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class FaceCompareDirectoriesApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1920x1200")
        self.root.minsize(1500, 950)
        self.root.configure(bg=WINDOW_BG)

        self.dir_a: Optional[str] = None
        self.dir_b: Optional[str] = None
        self.index_a: list[FaceResult] = []
        self.index_b: list[FaceResult] = []
        self.matches: list[MatchRecord] = []
        self.current_match_idx: int = -1
        self.tk_refs = []
        self.current_result_a: Optional[FaceResult] = None
        self.current_result_b: Optional[FaceResult] = None

        self.threshold_var = tk.StringVar(value=f"{DEFAULT_THRESHOLD:.2f}")
        self.frame_step_var = tk.StringVar(value="30")
        self.recursive_var = tk.BooleanVar(value=True)
        self.video_var = tk.BooleanVar(value=True)
        self.only_above_threshold_var = tk.BooleanVar(value=True)
        self.save_faces_var = tk.BooleanVar(value=False)
        self.use_cached_index_var = tk.BooleanVar(value=True)
        self.force_reindex_var = tk.BooleanVar(value=False)

        self.arcface = self._init_arcface()
        self._build_ui()

    def _init_arcface(self):
        if FaceAnalysis is None:
            raise RuntimeError(f"InsightFace non disponibile: {INSIGHTFACE_ERROR}")

        validate_local_insightface_models()
        root_dir = get_insightface_root()

        app = FaceAnalysis(
            name=INSIGHTFACE_MODEL_NAME,
            root=str(root_dir),
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, bg=WINDOW_BG)
        top.pack(fill="x", padx=12, pady=10)

        tk.Label(top, text=APP_TITLE, bg=WINDOW_BG, fg=TEXT_FG, font=("Helvetica", 18, "bold")).pack(anchor="w")
        tk.Label(
            top,
            text="Directory A vs B | immagini + video | tutti i volti | keypoint gialli con MediaPipe | embedding ArcFace per il match",
            bg=WINDOW_BG,
            fg=MUTED,
            font=("Helvetica", 10),
        ).pack(anchor="w", pady=(3, 0))

        controls = tk.Frame(self.root, bg=WINDOW_BG)
        controls.pack(fill="x", padx=12, pady=(0, 8))

        ttk.Button(controls, text="Directory A", command=self.select_dir_a).pack(side="left", padx=4)
        ttk.Button(controls, text="Directory B", command=self.select_dir_b).pack(side="left", padx=4)
        ttk.Button(controls, text="Indicizza A+B", command=self.index_both).pack(side="left", padx=4)
        ttk.Button(controls, text="Confronta", command=self.compare_indexes).pack(side="left", padx=4)
        ttk.Button(controls, text="Esporta HTML", command=self.export_html).pack(side="left", padx=4)
        ttk.Button(controls, text="Esporta CSV", command=self.export_csv).pack(side="left", padx=4)
        ttk.Button(controls, text="Esporta Pacchetto", command=self.export_package).pack(side="left", padx=4)
        ttk.Button(controls, text="Esci", command=self.root.destroy).pack(side="right", padx=4)

        options = tk.Frame(self.root, bg=WINDOW_BG)
        options.pack(fill="x", padx=12, pady=(0, 8))

        for text, var in [
            ("Sottodirectory", self.recursive_var),
            ("Processa video", self.video_var),
            ("Salva solo match >= soglia", self.only_above_threshold_var),
            ("Salva crop volti nei pacchetti", self.save_faces_var),
            ("Usa index PKL se presente", self.use_cached_index_var),
            ("Forza reindicizzazione", self.force_reindex_var),
        ]:
            tk.Checkbutton(
                options,
                text=text,
                variable=var,
                bg=WINDOW_BG,
                fg=TEXT_FG,
                selectcolor=PANEL_BG,
                activebackground=WINDOW_BG,
                activeforeground=TEXT_FG,
            ).pack(side="left", padx=6)

        tk.Label(options, text="Soglia:", bg=WINDOW_BG, fg=TEXT_FG).pack(side="left", padx=(16, 4))
        tk.Entry(options, textvariable=self.threshold_var, width=7, justify="center").pack(side="left", padx=4)
        tk.Label(options, text="Frame step video:", bg=WINDOW_BG, fg=TEXT_FG).pack(side="left", padx=(16, 4))
        tk.Entry(options, textvariable=self.frame_step_var, width=7, justify="center").pack(side="left", padx=4)

        paths = tk.Frame(self.root, bg=WINDOW_BG)
        paths.pack(fill="x", padx=12, pady=(0, 8))
        self.dir_a_label = tk.Label(paths, text="Directory A: non selezionata", bg=WINDOW_BG, fg=ACCENT, anchor="w")
        self.dir_a_label.pack(fill="x")
        self.dir_b_label = tk.Label(paths, text="Directory B: non selezionata", bg=WINDOW_BG, fg=ACCENT, anchor="w")
        self.dir_b_label.pack(fill="x")

        self.status_var = tk.StringVar(value="Seleziona le directory e indicizza.")
        tk.Label(self.root, textvariable=self.status_var, bg=WINDOW_BG, fg=TEXT_FG, anchor="w",
                 font=("Helvetica", 10, "bold")).pack(fill="x", padx=12, pady=(0, 4))

        prog = tk.Frame(self.root, bg=WINDOW_BG)
        prog.pack(fill="x", padx=12, pady=(0, 6))
        self.progress = ttk.Progressbar(prog, mode="determinate")
        self.progress.pack(fill="x", side="left", expand=True)
        self.progress_label = tk.Label(prog, text="0%", bg=WINDOW_BG, fg=MUTED, width=8)
        self.progress_label.pack(side="left", padx=8)

        score_frame = tk.Frame(self.root, bg=WINDOW_BG)
        score_frame.pack(fill="x", padx=12, pady=(0, 8))
        self.score_label = tk.Label(
            score_frame,
            text="Range teorico cosine similarity: min -1.00 | max +1.00 | confronto: ArcFace embedding",
            bg=WINDOW_BG,
            fg=TEXT_FG,
            anchor="w",
            font=("Helvetica", 10, "bold"),
        )
        self.score_label.pack(fill="x", pady=(0, 4))
        self.score_canvas = tk.Canvas(score_frame, width=1200, height=74, bg="#181818",
                                      highlightthickness=1, highlightbackground="#333333")
        self.score_canvas.pack(fill="x")

        nav = tk.Frame(self.root, bg=WINDOW_BG)
        nav.pack(fill="x", padx=12, pady=(0, 6))
        ttk.Button(nav, text="<< Precedente", command=self.show_prev_match).pack(side="left", padx=4)
        ttk.Button(nav, text="Successivo >>", command=self.show_next_match).pack(side="left", padx=4)
        self.match_nav_label = tk.Label(nav, text="Nessun match", bg=WINDOW_BG, fg=MUTED)
        self.match_nav_label.pack(side="left", padx=12)
        ttk.Button(nav, text="Apri originale A", command=lambda: self._open_current_external("Elemento A")).pack(side="right", padx=4)
        ttk.Button(nav, text="Apri originale B", command=lambda: self._open_current_external("Elemento B")).pack(side="right", padx=4)

        main = tk.Frame(self.root, bg=WINDOW_BG)
        main.pack(fill="both", expand=True, padx=10, pady=6)
        self.panel_left = self._make_side(main, "Elemento A")
        self.panel_left["frame"].pack(side="left", fill="both", expand=True, padx=6)
        self.panel_right = self._make_side(main, "Elemento B")
        self.panel_right["frame"].pack(side="left", fill="both", expand=True, padx=6)

        bottom = tk.Frame(self.root, bg=WINDOW_BG)
        bottom.pack(fill="both", expand=False, padx=12, pady=(0, 10))
        self.metrics = tk.Text(
            bottom,
            height=14,
            bg="#191919",
            fg=TEXT_FG,
            insertbackground=TEXT_FG,
            relief="flat",
            wrap="word",
            font=("Courier", 10),
        )
        self.metrics.pack(fill="both", expand=True)
        self._set_metrics("Pronto.\n")
        self._draw_score_bar(None)

    def _make_side(self, parent: tk.Widget, title: str):
        frame = tk.LabelFrame(parent, text=title, bg=WINDOW_BG, fg=TEXT_FG,
                              font=("Helvetica", 12, "bold"), padx=8, pady=8, bd=2)
        grid = tk.Frame(frame, bg=WINDOW_BG)
        grid.pack(fill="both", expand=True)

        original = self._make_image_box(grid, "Originale / Frame")
        original["frame"].grid(row=0, column=0, padx=6, pady=6, sticky="nsew")
        annotated = self._make_image_box(grid, "Volto selezionato + keypoint")
        annotated["frame"].grid(row=0, column=1, padx=6, pady=6, sticky="nsew")
        crop = self._make_image_box(grid, "Crop volto")
        crop["frame"].grid(row=1, column=0, padx=6, pady=6, sticky="nsew")

        info = tk.Text(grid, height=10, width=42, bg=PANEL_BG, fg=TEXT_FG, relief="flat",
                       wrap="word", font=("Courier", 10))
        info.grid(row=1, column=1, padx=6, pady=6, sticky="nsew")

        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)

        original["label"].bind("<Double-Button-1>", lambda e, side=title, key="original": self._open_current_image(side, key))
        annotated["label"].bind("<Double-Button-1>", lambda e, side=title, key="annotated": self._open_current_image(side, key))
        crop["label"].bind("<Double-Button-1>", lambda e, side=title, key="crop": self._open_current_image(side, key))

        return {"frame": frame, "original": original["label"], "annotated": annotated["label"],
                "crop": crop["label"], "info": info}

    def _make_image_box(self, parent: tk.Widget, title: str):
        frame = tk.LabelFrame(parent, text=title, bg=WINDOW_BG, fg=MUTED, padx=6, pady=6)
        canvas = tk.Canvas(frame, bg=PANEL_BG, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        return {"frame": frame, "label": canvas}

    def _set_metrics(self, text: str) -> None:
        self.metrics.delete("1.0", tk.END)
        self.metrics.insert("1.0", text)

    def _get_threshold(self) -> float:
        try:
            value = float(self.threshold_var.get().strip().replace(",", "."))
        except Exception:
            value = DEFAULT_THRESHOLD
        return max(0.0, min(1.0, value))

    def _get_frame_step(self) -> int:
        try:
            value = int(float(self.frame_step_var.get().strip().replace(",", ".")))
        except Exception:
            value = 30
        return max(1, value)

    def select_dir_a(self) -> None:
        path = filedialog.askdirectory(title="Seleziona Directory A")
        if path:
            self.dir_a = path
            self.dir_a_label.configure(text=f"Directory A: {path}")

    def select_dir_b(self) -> None:
        path = filedialog.askdirectory(title="Seleziona Directory B")
        if path:
            self.dir_b = path
            self.dir_b_label.configure(text=f"Directory B: {path}")

    def _update_progress(self, value: int, maximum: int, text: str) -> None:
        self.progress["maximum"] = max(1, maximum)
        self.progress["value"] = min(value, maximum)
        pct = int((min(value, maximum) / max(1, maximum)) * 100)
        self.progress_label.configure(text=f"{pct}%")
        self.status_var.set(text)
        self.root.update_idletasks()

    def _iter_files(self, root_dir: str) -> list[str]:
        base = Path(root_dir)
        files = []
        iterator = base.rglob("*") if self.recursive_var.get() else base.glob("*")
        for p in iterator:
            if p.is_file():
                ext = p.suffix.lower()
                if ext in IMAGE_EXTS or (self.video_var.get() and ext in VIDEO_EXTS):
                    files.append(str(p))
        return sorted(files)

    def _save_index_pkl(self, dir_path: str, items: list[FaceResult]) -> Path:
        pkl_path = make_index_path(dir_path)
        serializable = []
        for item in items:
            serializable.append(
                {
                    "source_path": item.source_path,
                    "source_type": item.source_type,
                    "frame_index": item.frame_index,
                    "timestamp_sec": item.timestamp_sec,
                    "face_index": item.face_index,
                    "total_faces_in_source": item.total_faces_in_source,
                    "display_name": item.display_name,
                    "original_bgr": item.original_bgr,
                    "annotated_bgr": item.annotated_bgr,
                    "crop_bgr": item.crop_bgr,
                    "embedding": item.embedding,
                    "similarity_ready": item.similarity_ready,
                    "face_count_arcface": item.face_count_arcface,
                    "mp_landmark_count": item.mp_landmark_count,
                    "bbox": item.bbox,
                    "warning": item.warning,
                    "md5": item.md5,
                    "sha256": item.sha256,
                }
            )
        with open(pkl_path, "wb") as f:
            pickle.dump(serializable, f)
        return pkl_path

    def _load_index_pkl(self, dir_path: str) -> Optional[list[FaceResult]]:
        pkl_path = make_index_path(dir_path)
        if not pkl_path.exists():
            return None
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        results: list[FaceResult] = []
        for row in data:
            results.append(
                FaceResult(
                    source_path=row["source_path"],
                    source_type=row["source_type"],
                    frame_index=row["frame_index"],
                    timestamp_sec=row["timestamp_sec"],
                    face_index=row.get("face_index", 1),
                    total_faces_in_source=row.get("total_faces_in_source", 1),
                    display_name=row["display_name"],
                    original_bgr=row["original_bgr"],
                    annotated_bgr=row["annotated_bgr"],
                    crop_bgr=row["crop_bgr"],
                    embedding=row["embedding"],
                    similarity_ready=row["similarity_ready"],
                    face_count_arcface=row["face_count_arcface"],
                    mp_landmark_count=row["mp_landmark_count"],
                    bbox=row["bbox"],
                    warning=row["warning"],
                    md5=row["md5"],
                    sha256=row["sha256"],
                )
            )
        return results

    def index_both(self) -> None:
        try:
            if not self.dir_a or not self.dir_b:
                messagebox.showwarning("Attenzione", "Seleziona entrambe le directory.")
                return

            self.index_a = []
            self.index_b = []
            self.matches = []
            self.current_match_idx = -1
            self._draw_score_bar(None)

            loaded_a = False
            loaded_b = False

            if self.use_cached_index_var.get() and not self.force_reindex_var.get():
                cached_a = self._load_index_pkl(self.dir_a)
                if cached_a is not None:
                    self.index_a = cached_a
                    loaded_a = True
                cached_b = self._load_index_pkl(self.dir_b)
                if cached_b is not None:
                    self.index_b = cached_b
                    loaded_b = True

            if not loaded_a:
                files_a = self._iter_files(self.dir_a)
                tmp_a: list[FaceResult] = []
                for i, f in enumerate(files_a, start=1):
                    tmp_a.extend(self._process_path(f))
                    self._update_progress(i, max(1, len(files_a)), f"Indicizzazione A: {Path(f).name}")
                self.index_a = tmp_a
                pkl_a = self._save_index_pkl(self.dir_a, self.index_a)
            else:
                pkl_a = make_index_path(self.dir_a)

            if not loaded_b:
                files_b = self._iter_files(self.dir_b)
                tmp_b: list[FaceResult] = []
                for i, f in enumerate(files_b, start=1):
                    tmp_b.extend(self._process_path(f))
                    self._update_progress(i, max(1, len(files_b)), f"Indicizzazione B: {Path(f).name}")
                self.index_b = tmp_b
                pkl_b = self._save_index_pkl(self.dir_b, self.index_b)
            else:
                pkl_b = make_index_path(self.dir_b)

            txt = []
            txt.append("=== INDICIZZAZIONE COMPLETATA ===")
            txt.append(f"Directory A: {self.dir_a}")
            txt.append(f"Volti indicizzati A: {len(self.index_a)}")
            txt.append(f"Index PKL A: {pkl_a}")
            txt.append(f"Caricato da cache A: {'Si' if loaded_a else 'No'}")
            txt.append("")
            txt.append(f"Directory B: {self.dir_b}")
            txt.append(f"Volti indicizzati B: {len(self.index_b)}")
            txt.append(f"Index PKL B: {pkl_b}")
            txt.append(f"Caricato da cache B: {'Si' if loaded_b else 'No'}")
            txt.append("")
            txt.append(f"Frame step video: {self._get_frame_step()}")
            txt.append(f"Sottodirectory: {'Si' if self.recursive_var.get() else 'No'}")
            txt.append(f"Video: {'Si' if self.video_var.get() else 'No'}")
            txt.append("Modalita multi-volto: attiva")
            self._set_metrics("\n".join(txt))
            self.status_var.set("Indicizzazione multi-volto completata. PKL salvati/caricati correttamente.")
            self._update_progress(1, 1, "Indicizzazione completata")

        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Errore", f"Errore in indicizzazione:\n{exc}")

    def _process_path(self, path: str) -> list[FaceResult]:
        ext = Path(path).suffix.lower()
        if ext in IMAGE_EXTS:
            return self._process_image(path)
        if self.video_var.get() and ext in VIDEO_EXTS:
            return self._process_video(path)
        return []

    def _process_image(self, path: str) -> list[FaceResult]:
        bgr = cv2.imread(path)
        if bgr is None:
            return []
        return self._extract_faces_from_bgr(
            bgr=bgr,
            source_path=path,
            source_type="image",
            frame_index=0,
            timestamp_sec=0.0,
            display_prefix=Path(path).name,
            source_hash_path=path,
        )

    def _process_video(self, path: str) -> list[FaceResult]:
        results: list[FaceResult] = []
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return results

        frame_step = self._get_frame_step()
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 0.0

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_step == 0:
                ts = (idx / fps) if fps > 0 else 0.0
                results.extend(
                    self._extract_faces_from_bgr(
                        bgr=frame,
                        source_path=path,
                        source_type="video",
                        frame_index=idx,
                        timestamp_sec=ts,
                        display_prefix=f"{Path(path).name} | frame {idx} | t={ts:.2f}s",
                        source_hash_path=path,
                    )
                )
            idx += 1

        cap.release()
        return results

    def _extract_faces_from_bgr(
        self,
        bgr: np.ndarray,
        source_path: str,
        source_type: str,
        frame_index: int,
        timestamp_sec: float,
        display_prefix: str,
        source_hash_path: str,
    ) -> list[FaceResult]:
        arc_faces = self.arcface.get(bgr)
        total_faces = len(arc_faces)
        if total_faces == 0:
            return []

        h, w = bgr.shape[:2]
        results: list[FaceResult] = []

        for face_idx, face in enumerate(sorted(arc_faces, key=lambda f: area_from_bbox(f.bbox), reverse=True), start=1):
            annotated = bgr.copy()
            crop = None
            warning = ""
            mp_landmark_count = 0
            embedding = None
            similarity_ready = False

            abox = face.bbox.astype(int)
            x1, y1, x2, y2 = map(int, abox)
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, w)
            y2 = min(y2, h)
            bbox = (x1, y1, x2, y2)

            if y2 > y1 and x2 > x1:
                crop = bgr[y1:y2, x1:x2].copy()
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                with mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                ) as face_mesh:
                    mp_result = face_mesh.process(crop_rgb)

                if mp_result.multi_face_landmarks:
                    face_landmarks = mp_result.multi_face_landmarks[0]
                    mp_landmark_count = len(face_landmarks.landmark)
                    ch, cw = crop.shape[:2]
                    for lm in face_landmarks.landmark:
                        cx = int(lm.x * cw)
                        cy = int(lm.y * ch)
                        cv2.circle(annotated, (x1 + cx, y1 + cy), 1, YELLOW_BGR, -1, lineType=cv2.LINE_AA)
                else:
                    warning = join_notes(warning, "MediaPipe non ha rilevato landmark per questo volto.")
            else:
                warning = join_notes(warning, "Crop volto non disponibile.")

            cv2.rectangle(annotated, (x1, y1), (x2, y2), YELLOW_BGR, 2)
            cv2.putText(
                annotated,
                f"Face {face_idx}/{total_faces}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                YELLOW_BGR,
                2,
                lineType=cv2.LINE_AA,
            )

            emb = getattr(face, "embedding", None)
            if emb is not None:
                embedding = np.asarray(emb, dtype=np.float32)
                similarity_ready = True

            if not similarity_ready:
                continue

            display_name = f"{display_prefix} | volto {face_idx}/{total_faces}"
            results.append(
                FaceResult(
                    source_path=source_path,
                    source_type=source_type,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    face_index=face_idx,
                    total_faces_in_source=total_faces,
                    display_name=display_name,
                    original_bgr=bgr,
                    annotated_bgr=annotated,
                    crop_bgr=crop,
                    embedding=embedding,
                    similarity_ready=True,
                    face_count_arcface=total_faces,
                    mp_landmark_count=mp_landmark_count,
                    bbox=bbox,
                    warning=warning,
                    md5=file_md5(source_hash_path),
                    sha256=file_sha256(source_hash_path),
                )
            )

        return results

    def compare_indexes(self) -> None:
        try:
            if not self.index_a or not self.index_b:
                messagebox.showwarning("Attenzione", "Indicizza prima le due directory.")
                return

            self.matches = []
            self.current_match_idx = -1
            threshold = self._get_threshold()
            total = len(self.index_a) * len(self.index_b)
            done = 0

            for a in self.index_a:
                for b in self.index_b:
                    done += 1
                    score = cosine_similarity(a.embedding, b.embedding)
                    verdict = self._interpret_similarity(score)
                    if (not self.only_above_threshold_var.get()) or (score >= threshold):
                        self.matches.append(
                            MatchRecord(
                                score=score,
                                verdict=verdict,
                                source_a=a.source_path,
                                source_b=b.source_path,
                                display_a=a.display_name,
                                display_b=b.display_name,
                                type_a=a.source_type,
                                type_b=b.source_type,
                                frame_a=a.frame_index,
                                frame_b=b.frame_index,
                                time_a=a.timestamp_sec,
                                time_b=b.timestamp_sec,
                                face_idx_a=a.face_index,
                                face_idx_b=b.face_index,
                                total_faces_a=a.total_faces_in_source,
                                total_faces_b=b.total_faces_in_source,
                                md5_a=a.md5,
                                md5_b=b.md5,
                            )
                        )

                    if done % 100 == 0 or done == total:
                        self._update_progress(done, total, f"Confronto in corso: {done}/{total}")

            self.matches.sort(key=lambda m: m.score, reverse=True)

            txt = []
            txt.append("=== CONFRONTO MULTI-VOLTO COMPLETATO ===")
            txt.append(f"Volti indicizzati A: {len(self.index_a)}")
            txt.append(f"Volti indicizzati B: {len(self.index_b)}")
            txt.append(f"Confronti totali teorici: {total}")
            txt.append(f"Match salvati: {len(self.matches)}")
            txt.append(f"Soglia: {threshold:.2f}")
            txt.append(f"Solo sopra soglia: {'Si' if self.only_above_threshold_var.get() else 'No'}")
            if self.matches:
                top = self.matches[0]
                txt.append("")
                txt.append("Miglior match:")
                txt.append(f"Score: {top.score:.4f}")
                txt.append(f"A: {top.display_a}")
                txt.append(f"B: {top.display_b}")
                self._set_metrics("\n".join(txt))
                self.current_match_idx = 0
                self._show_match(0)
            else:
                txt.append("")
                txt.append("Nessun match salvato con i criteri correnti.")
                self._set_metrics("\n".join(txt))
                self._draw_score_bar(None)
                self.match_nav_label.configure(text="Nessun match")

            self.status_var.set("Confronto multi-volto completato.")

        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Errore", f"Errore nel confronto:\n{exc}")

    def _interpret_similarity(self, score: float) -> str:
        threshold = self._get_threshold()
        medium = max(0.0, threshold - 0.20)
        if score >= threshold:
            return "Compatibilita alta"
        if score >= medium:
            return "Compatibilita media - valutare con cautela"
        return "Compatibilita bassa"

    def _find_result(self, source_path: str, frame_index: int, source_type: str, face_index: int, side: str) -> Optional[FaceResult]:
        arr = self.index_a if side == "A" else self.index_b
        for item in arr:
            if (
                item.source_path == source_path
                and item.frame_index == frame_index
                and item.source_type == source_type
                and item.face_index == face_index
            ):
                return item
        return None

    def _show_match(self, idx: int) -> None:
        if not self.matches:
            return

        idx = max(0, min(idx, len(self.matches) - 1))
        self.current_match_idx = idx
        m = self.matches[idx]

        ra = self._find_result(m.source_a, m.frame_a, m.type_a, m.face_idx_a, "A")
        rb = self._find_result(m.source_b, m.frame_b, m.type_b, m.face_idx_b, "B")

        if ra:
            self._render_result(self.panel_left, ra)
        if rb:
            self._render_result(self.panel_right, rb)

        self._draw_score_bar(m.score)
        self.match_nav_label.configure(text=f"Match {idx + 1}/{len(self.matches)} | score {m.score:.4f}")

        lines = []
        lines.append("=== MATCH CORRENTE ===")
        lines.append(f"Indice: {idx + 1}/{len(self.matches)}")
        lines.append(f"Score ArcFace: {m.score:.4f}")
        lines.append(f"Valutazione: {m.verdict}")
        lines.append("")
        lines.append("Confronto biometrico:")
        lines.append("- ArcFace / InsightFace per estrazione embedding")
        lines.append("- cosine similarity per il punteggio")
        lines.append("- MediaPipe usato per keypoint gialli del volto selezionato")
        lines.append("")
        lines.append(f"A: {m.display_a}")
        lines.append(f"   File: {m.source_a}")
        lines.append(f"   Tipo: {m.type_a}")
        lines.append(f"   Frame: {m.frame_a}")
        lines.append(f"   Timestamp: {m.time_a:.2f}s")
        lines.append(f"   Volto: {m.face_idx_a}/{m.total_faces_a}")
        lines.append(f"   MD5: {m.md5_a}")
        lines.append("")
        lines.append(f"B: {m.display_b}")
        lines.append(f"   File: {m.source_b}")
        lines.append(f"   Tipo: {m.type_b}")
        lines.append(f"   Frame: {m.frame_b}")
        lines.append(f"   Timestamp: {m.time_b:.2f}s")
        lines.append(f"   Volto: {m.face_idx_b}/{m.total_faces_b}")
        lines.append(f"   MD5: {m.md5_b}")
        self._set_metrics("\n".join(lines))

    def show_prev_match(self) -> None:
        if self.matches:
            self._show_match(self.current_match_idx - 1 if self.current_match_idx > 0 else 0)

    def show_next_match(self) -> None:
        if self.matches:
            self._show_match(self.current_match_idx + 1 if self.current_match_idx < len(self.matches) - 1 else self.current_match_idx)

    def _render_result(self, panel, result: FaceResult) -> None:
        if panel is self.panel_left:
            self.current_result_a = result
        elif panel is self.panel_right:
            self.current_result_b = result

        self._set_label_image(panel["original"], result.original_bgr, PREVIEW_SIZE)
        self._set_label_image(panel["annotated"], result.annotated_bgr, PREVIEW_SIZE)
        self._set_label_image(panel["crop"], result.crop_bgr, CROP_SIZE)

        info_lines = [
            f"Display: {result.display_name}",
            f"Tipo: {result.source_type}",
            f"File: {os.path.basename(result.source_path)}",
            f"Frame: {result.frame_index}",
            f"Timestamp: {result.timestamp_sec:.2f}s",
            f"Volto: {result.face_index}/{result.total_faces_in_source}",
            f"Volti rilevati: {result.face_count_arcface}",
            f"Landmark MediaPipe: {result.mp_landmark_count}",
            f"BBox: {result.bbox if result.bbox else 'Nessuna'}",
            f"Embedding ArcFace: {'Si' if result.similarity_ready else 'No'}",
            f"MD5: {result.md5}",
            f"SHA256: {result.sha256}",
            f"Avvisi: {result.warning or 'Nessuno'}",
        ]
        panel["info"].delete("1.0", tk.END)
        panel["info"].insert("1.0", "\n".join(info_lines))

    def _open_current_image(self, side: str, key: str) -> None:
        result = self.current_result_a if side == "Elemento A" else self.current_result_b
        if result is None:
            return

        if key == "original":
            bgr = result.original_bgr
        elif key == "annotated":
            bgr = result.annotated_bgr
        else:
            bgr = result.crop_bgr

        self._open_image_popup(f"{side} - {key}", bgr)

    def _open_current_external(self, side: str) -> None:
        result = self.current_result_a if side == "Elemento A" else self.current_result_b
        if result is None:
            return
        self._open_external_path(result.source_path)

    def _open_external_path(self, path: str) -> None:
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            messagebox.showerror("Errore", f"Impossibile aprire il file nel viewer di sistema:\n{exc}")

    def _open_image_popup(self, title: str, bgr: Optional[np.ndarray]) -> None:
        if bgr is None:
            return

        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("1400x900")
        win.configure(bg=WINDOW_BG)

        toolbar = tk.Frame(win, bg=WINDOW_BG)
        toolbar.pack(fill="x", padx=8, pady=8)

        canvas = tk.Canvas(win, bg="#000000", highlightthickness=0)
        hbar = tk.Scrollbar(win, orient="horizontal", command=canvas.xview)
        vbar = tk.Scrollbar(win, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        hbar.pack(side="bottom", fill="x")
        vbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        original_pil = Image.fromarray(rgb)
        state = {"scale": 1.0, "photo": None}

        def render():
            scale = state["scale"]
            w = max(1, int(original_pil.width * scale))
            h = max(1, int(original_pil.height * scale))
            resized = original_pil.resize((w, h), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            state["photo"] = photo
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=photo)
            canvas.config(scrollregion=(0, 0, w, h))

        def zoom_in():
            state["scale"] *= 1.25
            render()

        def zoom_out():
            state["scale"] /= 1.25
            render()

        def reset_zoom():
            state["scale"] = 1.0
            render()

        ttk.Button(toolbar, text="Zoom +", command=zoom_in).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Zoom -", command=zoom_out).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Reset", command=reset_zoom).pack(side="left", padx=4)

        temp_path = Path.cwd() / "_facecompare_popup_preview.png"
        def open_in_system_viewer():
            try:
                cv2.imwrite(str(temp_path), bgr)
                self._open_external_path(str(temp_path))
            except Exception as exc:
                messagebox.showerror("Errore", f"Impossibile aprire il viewer di sistema:\n{exc}")

        ttk.Button(toolbar, text="Apri nel viewer di sistema", command=open_in_system_viewer).pack(side="left", padx=8)

        def on_mousewheel(event):
            if event.state & 0x4:
                if event.delta > 0:
                    zoom_in()
                else:
                    zoom_out()
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<MouseWheel>", on_mousewheel)
        render()

    def _set_label_image(self, widget, bgr: Optional[np.ndarray], size: Tuple[int, int]) -> None:
        if bgr is None:
            widget.delete("all")
            widget.create_text(
                20, 20,
                anchor="nw",
                text="Non disponibile",
                fill=MUTED,
                font=("Helvetica", 12, "bold"),
            )
            widget._source_bgr = None
            return

        widget._source_bgr = bgr

        def render(event=None):
            current = getattr(widget, "_source_bgr", None)
            if current is None:
                return

            cw = max(widget.winfo_width(), 50)
            ch = max(widget.winfo_height(), 50)

            rgb = cv2.cvtColor(current, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            scale = min(cw / pil.width, ch / pil.height)
            nw = max(1, int(pil.width * scale))
            nh = max(1, int(pil.height * scale))

            resized = pil.resize((nw, nh), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized)

            self.tk_refs.append(photo)
            widget.delete("all")
            x = (cw - nw) // 2
            y = (ch - nh) // 2
            widget.create_image(x, y, anchor="nw", image=photo)
            widget.image = photo

        if not getattr(widget, "_bind_resize_done", False):
            widget.bind("<Configure>", render)
            widget._bind_resize_done = True

        render()
    def _draw_score_bar(self, similarity: Optional[float]) -> None:
        canvas = self.score_canvas
        canvas.delete("all")
        w = max(canvas.winfo_width(), 1200)
        left, right, top, bottom = 60, w - 60, 24, 48
        mid_y = (top + bottom) // 2

        canvas.create_rectangle(left, top, right, bottom, fill="#262626", outline="#444444")

        def x_from_score(score: float) -> float:
            s = max(-1.0, min(1.0, score))
            return left + ((s + 1.0) / 2.0) * (right - left)

        for val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            x = x_from_score(val)
            canvas.create_line(x, top, x, bottom, fill="#666666")
            canvas.create_text(x, bottom + 14, text=f"{val:+.2f}", fill=TEXT_FG, font=("Helvetica", 9))

        x0 = x_from_score(0.0)
        canvas.create_line(x0, top - 4, x0, bottom + 4, fill="#999999", width=2)

        threshold = self._get_threshold()
        xthr = x_from_score(threshold)
        canvas.create_line(xthr, top - 8, xthr, bottom + 8, fill=ACCENT, width=3)
        canvas.create_text(xthr, top - 12, text=f"soglia {threshold:.2f}", fill=ACCENT, font=("Helvetica", 9, "bold"))

        if similarity is not None:
            xpos = x_from_score(similarity)
            if similarity >= threshold:
                color, verdict = GOOD, "Compatibilita alta"
            elif similarity >= max(0.0, threshold - 0.20):
                color, verdict = MID, "Compatibilita media"
            else:
                color, verdict = BAD, "Compatibilita bassa"
            fill_left = min(left, xpos)
            fill_right = max(left, xpos)
            canvas.create_rectangle(fill_left, top, fill_right, bottom, fill=color, outline="")
            canvas.create_oval(xpos - 8, mid_y - 8, xpos + 8, mid_y + 8, fill="#ffffff", outline="#000000")
            canvas.create_text(
                left,
                top - 12,
                text=f"score corrente: {similarity:.4f} | range teorico -1.00 .. +1.00 | {verdict}",
                anchor="w",
                fill=TEXT_FG,
                font=("Helvetica", 10, "bold"),
            )
        else:
            canvas.create_text(
                left,
                top - 12,
                text="Nessun punteggio visibile. Il confronto usa ArcFace embedding + cosine similarity.",
                anchor="w",
                fill=TEXT_FG,
                font=("Helvetica", 10, "bold"),
            )

        self.score_label.configure(
            text=f"Range teorico cosine similarity: min -1.00 | max +1.00 | soglia corrente: {threshold:.2f} | confronto: ArcFace embedding"
        )

    def export_csv(self) -> None:
        try:
            if not self.matches:
                messagebox.showwarning("Attenzione", "Esegui prima il confronto.")
                return
            save_path = filedialog.asksaveasfilename(
                title="Salva report CSV",
                defaultextension=".csv",
                initialfile=f"face_compare_dirs_multiface_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                filetypes=[("CSV", "*.csv")],
            )
            if not save_path:
                return
            with open(save_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow([
                    "rank", "score_arcface", "verdict",
                    "source_a", "display_a", "type_a", "frame_a", "time_a", "face_idx_a", "total_faces_a",
                    "source_b", "display_b", "type_b", "frame_b", "time_b", "face_idx_b", "total_faces_b",
                    "md5_a", "md5_b"
                ])
                for i, m in enumerate(self.matches, start=1):
                    writer.writerow([
                        i, f"{m.score:.6f}", m.verdict,
                        m.source_a, m.display_a, m.type_a, m.frame_a, f"{m.time_a:.2f}", m.face_idx_a, m.total_faces_a,
                        m.source_b, m.display_b, m.type_b, m.frame_b, f"{m.time_b:.2f}", m.face_idx_b, m.total_faces_b,
                        m.md5_a, m.md5_b
                    ])
            messagebox.showinfo("Completato", f"CSV salvato in:\n{save_path}")
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Errore", f"Impossibile esportare CSV:\n{exc}")

    def export_html(self) -> None:
        try:
            if not self.matches:
                messagebox.showwarning("Attenzione", "Esegui prima il confronto.")
                return
            save_path = filedialog.asksaveasfilename(
                title="Salva report HTML",
                defaultextension=".html",
                initialfile=f"face_compare_dirs_multiface_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                filetypes=[("HTML", "*.html")],
            )
            if not save_path:
                return
            Path(save_path).write_text(self._build_html(), encoding="utf-8")
            messagebox.showinfo("Completato", f"HTML salvato in:\n{save_path}")
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Errore", f"Impossibile esportare HTML:\n{exc}")

    def export_package(self) -> None:
        try:
            if not self.matches:
                messagebox.showwarning("Attenzione", "Esegui prima il confronto.")
                return
            base_dir = filedialog.askdirectory(title="Seleziona directory di destinazione")
            if not base_dir:
                return

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path(base_dir) / EXPORT_DIRNAME / f"compare_{ts}"
            out_dir.mkdir(parents=True, exist_ok=True)

            html_path = out_dir / "report.html"
            csv_path = out_dir / "report.csv"
            faces_dir = out_dir / "faces"
            if self.save_faces_var.get():
                faces_dir.mkdir(exist_ok=True)

            html_path.write_text(self._build_html(), encoding="utf-8")

            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(["rank", "score_arcface", "verdict", "source_a", "display_a", "source_b", "display_b"])
                for i, m in enumerate(self.matches, start=1):
                    writer.writerow([i, f"{m.score:.6f}", m.verdict, m.source_a, m.display_a, m.source_b, m.display_b])

            if self.save_faces_var.get():
                for idx, m in enumerate(self.matches[:300], start=1):
                    ra = self._find_result(m.source_a, m.frame_a, m.type_a, m.face_idx_a, "A")
                    rb = self._find_result(m.source_b, m.frame_b, m.type_b, m.face_idx_b, "B")
                    if ra and ra.crop_bgr is not None:
                        cv2.imwrite(str(faces_dir / f"{idx:04d}_A.png"), ra.crop_bgr)
                    if rb and rb.crop_bgr is not None:
                        cv2.imwrite(str(faces_dir / f"{idx:04d}_B.png"), rb.crop_bgr)

            self._append_history_csv(out_dir)
            messagebox.showinfo("Completato", f"Pacchetto esportato in:\n{out_dir}")
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Errore", f"Impossibile esportare pacchetto:\n{exc}")

    def _append_history_csv(self, out_dir: Path) -> None:
        history_path = out_dir.parent / HISTORY_CSV
        header = ["timestamp", "dir_a", "dir_b", "threshold", "count_a", "count_b", "matches", "export_dir"]
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            self.dir_a or "",
            self.dir_b or "",
            f"{self._get_threshold():.2f}",
            len(self.index_a),
            len(self.index_b),
            len(self.matches),
            str(out_dir),
        ]
        append_csv_row(history_path, header, row)

    def _build_html(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        items = []
        for i, m in enumerate(self.matches[:300], start=1):
            ra = self._find_result(m.source_a, m.frame_a, m.type_a, m.face_idx_a, "A")
            rb = self._find_result(m.source_b, m.frame_b, m.type_b, m.face_idx_b, "B")
            a_img = img_to_b64(ra.annotated_bgr if ra else None)
            b_img = img_to_b64(rb.annotated_bgr if rb else None)
            items.append(f"""
<div class='card'>
<h3>Rank {i} | Score {m.score:.4f} | {m.verdict}</h3>
<div class='grid'>
<div>
<p><b>A</b><br>{escape_html(m.display_a)}<br><span class='small'>{escape_html(m.source_a)}</span></p>
<img src='data:image/png;base64,{a_img}'>
</div>
<div>
<p><b>B</b><br>{escape_html(m.display_b)}<br><span class='small'>{escape_html(m.source_b)}</span></p>
<img src='data:image/png;base64,{b_img}'>
</div>
</div>
<pre>Tipo A: {m.type_a} | frame A: {m.frame_a} | timestamp A: {m.time_a:.2f}s | volto A: {m.face_idx_a}/{m.total_faces_a}
Tipo B: {m.type_b} | frame B: {m.frame_b} | timestamp B: {m.time_b:.2f}s | volto B: {m.face_idx_b}/{m.total_faces_b}
MD5 A: {m.md5_a}
MD5 B: {m.md5_b}</pre>
</div>""")

        return f"""<!DOCTYPE html>
<html lang='it'>
<head>
<meta charset='utf-8'>
<title>Face Compare Directories v4 MultiFace Report</title>
<style>
body {{ background:#111; color:#f0f0f0; font-family:Arial,Helvetica,sans-serif; margin:20px; }}
.card {{ background:#1b1b1b; border:1px solid #333; border-radius:12px; padding:16px; margin-bottom:18px; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
img {{ max-width:100%; border-radius:8px; border:1px solid #444; }}
.h {{ color:#ffd400; }}
.small {{ color:#bbb; font-size:12px; }}
pre {{ white-space:pre-wrap; background:#161616; padding:12px; border-radius:8px; }}
</style>
</head>
<body>
<h1 class='h'>Face Compare Directories v4 - MultiFace</h1>
<p class='small'>Generato il {now}</p>

<div class='card'>
<h2>Sintesi</h2>
<pre>Directory A: {escape_html(self.dir_a or '')}
Directory B: {escape_html(self.dir_b or '')}
Volti indicizzati A: {len(self.index_a)}
Volti indicizzati B: {len(self.index_b)}
Match salvati: {len(self.matches)}
Soglia: {self._get_threshold():.2f}
Video abilitati: {'Si' if self.video_var.get() else 'No'}
Frame step: {self._get_frame_step()}
Modalita multi-volto: attiva</pre>
</div>

<div class='card'>
<h2>Spiegazione tecnica</h2>
<pre>Il confronto biometrico usa ArcFace / InsightFace per estrarre gli embedding facciali.
Il punteggio finale e calcolato con cosine similarity tra i due embedding.

Questa versione indicizza tutti i volti trovati nelle immagini e nei frame video campionati.
Ogni volto viene salvato come elemento autonomo con:
- indice volto
- numero totale di volti nella sorgente
- crop volto
- bounding box
- embedding ArcFace

MediaPipe viene usato per rilevare e disegnare i keypoint gialli del volto selezionato.
ArcFace viene usato per il confronto biometrico vero e proprio.</pre>
</div>

{''.join(items)}
</body>
</html>"""


def main() -> None:
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    FaceCompareDirectoriesApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
