#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import base64
import csv
import hashlib
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from shutil import copy2
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

APP_TITLE = "Face Compare Multi-Face - By Visi@n antonio@broi.it"
DEFAULT_THRESHOLD = 0.55
EXPORT_DIRNAME = "face_compare_exports_multi"
HISTORY_CSV = "face_compare_history_multi.csv"
WINDOW_BG = "#111111"
PANEL_BG = "#1b1b1b"
TEXT_FG = "#f3f3f3"
MUTED = "#bbbbbb"
ACCENT = "#ffd400"
GOOD = "#1db954"
MID = "#f4c542"
BAD = "#e74c3c"
PREVIEW_SIZE = (800, 600)
CROP_SIZE = (320, 320)
YELLOW_BGR = (0, 255, 255)


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

class DetectedFace:
    image_path: str
    face_index: int
    original_bgr: np.ndarray
    annotated_bgr: np.ndarray
    crop_bgr: Optional[np.ndarray]
    embedding: Optional[np.ndarray]
    similarity_ready: bool
    total_faces_arcface: int
    mp_landmark_count: int
    bbox: Optional[Tuple[int, int, int, int]]
    warning: str
    md5: str
    sha256: str


@dataclass
class MatchRecord:
    idx_a: int
    idx_b: int
    score: float
    verdict: str


class FaceCompareMultiApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1920x1200")
        self.root.minsize(1600, 980)
        self.root.configure(bg=WINDOW_BG)

        self.path_a: Optional[str] = None
        self.path_b: Optional[str] = None
        self.faces_a: list[DetectedFace] = []
        self.faces_b: list[DetectedFace] = []
        self.matches: list[MatchRecord] = []
        self.current_match_idx = -1
        self.tk_refs: list[ImageTk.PhotoImage] = []
        self.threshold_var = tk.StringVar(value=f"{DEFAULT_THRESHOLD:.2f}")

        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
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
        top.pack(fill="x", padx=12, pady=12)

        tk.Label(top, text=APP_TITLE, bg=WINDOW_BG, fg=TEXT_FG, font=("Helvetica", 18, "bold")).pack(anchor="w")
        tk.Label(
            top,
            text="Comparazione multi-volto: tutti i volti di A contro tutti i volti di B | MediaPipe per keypoint gialli | ArcFace per embedding",
            bg=WINDOW_BG,
            fg=MUTED,
            font=("Helvetica", 10),
        ).pack(anchor="w", pady=(4, 0))

        controls = tk.Frame(self.root, bg=WINDOW_BG)
        controls.pack(fill="x", padx=12, pady=(0, 8))
        ttk.Button(controls, text="Apri Immagine A", command=self.load_a).pack(side="left", padx=4)
        ttk.Button(controls, text="Apri Immagine B", command=self.load_b).pack(side="left", padx=4)
        ttk.Button(controls, text="Confronta", command=self.compare).pack(side="left", padx=4)
        ttk.Button(controls, text="Esporta HTML", command=self.export_html).pack(side="left", padx=4)
        ttk.Button(controls, text="Esporta CSV", command=self.export_csv).pack(side="left", padx=4)
        ttk.Button(controls, text="Esporta Pacchetto", command=self.export_package).pack(side="left", padx=4)
        tk.Label(controls, text="Soglia:", bg=WINDOW_BG, fg=TEXT_FG).pack(side="left", padx=(18, 4))
        tk.Entry(controls, textvariable=self.threshold_var, width=8, justify="center").pack(side="left", padx=4)
        ttk.Button(controls, text="Esci", command=self.root.destroy).pack(side="right", padx=4)

        nav = tk.Frame(self.root, bg=WINDOW_BG)
        nav.pack(fill="x", padx=12, pady=(0, 8))
        ttk.Button(nav, text="<< Precedente", command=self.prev_match).pack(side="left", padx=4)
        ttk.Button(nav, text="Successivo >>", command=self.next_match).pack(side="left", padx=4)
        self.nav_label = tk.Label(nav, text="Nessun match", bg=WINDOW_BG, fg=ACCENT, anchor="w", font=("Helvetica", 11, "bold"))
        self.nav_label.pack(side="left", padx=12)

        self.status_var = tk.StringVar(value="Carica due immagini e premi Confronta.")
        tk.Label(
            self.root,
            textvariable=self.status_var,
            bg=WINDOW_BG,
            fg=ACCENT,
            anchor="w",
            justify="left",
            font=("Helvetica", 11, "bold"),
        ).pack(fill="x", padx=14, pady=(0, 8))

        score = tk.Frame(self.root, bg=WINDOW_BG)
        score.pack(fill="x", padx=12, pady=(0, 6))
        self.score_label = tk.Label(
            score,
            text="Range teorico cosine similarity: min -1.00 | max +1.00 | confronto: ArcFace embedding",
            bg=WINDOW_BG, fg=TEXT_FG, anchor="w", font=("Helvetica", 10, "bold"),
        )
        self.score_label.pack(fill="x", pady=(0, 4))
        self.score_canvas = tk.Canvas(score, width=1200, height=74, bg="#181818", highlightthickness=1, highlightbackground="#333333")
        self.score_canvas.pack(fill="x")

        main = tk.Frame(self.root, bg=WINDOW_BG)
        main.pack(fill="both", expand=True, padx=10, pady=6)
        self.panel_a = self._make_side(main, "Immagine A")
        self.panel_a["frame"].pack(side="left", fill="both", expand=True, padx=6)
        self.panel_b = self._make_side(main, "Immagine B")
        self.panel_b["frame"].pack(side="left", fill="both", expand=True, padx=6)

        bottom = tk.Frame(self.root, bg=WINDOW_BG)
        bottom.pack(fill="both", expand=False, padx=12, pady=(0, 12))
        self.metrics = tk.Text(
            bottom,
            height=14,
            bg="#191919",
            fg=TEXT_FG,
            insertbackground=TEXT_FG,
            relief="flat",
            wrap="word",
            font=("Courier", 11),
        )
        self.metrics.pack(fill="both", expand=True)
        self._set_metrics("Pronto.\n")
        self._draw_score_bar(None)

    def _make_side(self, parent: tk.Widget, title: str):
        frame = tk.LabelFrame(parent, text=title, bg=WINDOW_BG, fg=TEXT_FG, font=("Helvetica", 12, "bold"), padx=8, pady=8, bd=2)
        grid = tk.Frame(frame, bg=WINDOW_BG)
        grid.pack(fill="both", expand=True)

        original = self._make_image_box(grid, "Originale")
        original["frame"].grid(row=0, column=0, padx=6, pady=6, sticky="nsew")
        annotated = self._make_image_box(grid, "Volto selezionato con keypoint gialli")
        annotated["frame"].grid(row=0, column=1, padx=6, pady=6, sticky="nsew")
        crop = self._make_image_box(grid, "Crop volto")
        crop["frame"].grid(row=1, column=0, padx=6, pady=6, sticky="nsew")
        info = tk.Text(grid, height=10, width=42, bg=PANEL_BG, fg=TEXT_FG, relief="flat", wrap="word", font=("Courier", 10))
        info.grid(row=1, column=1, padx=6, pady=6, sticky="nsew")

        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)
        return {"frame": frame, "original": original["label"], "annotated": annotated["label"], "crop": crop["label"], "info": info}

    def _make_image_box(self, parent: tk.Widget, title: str):
        frame = tk.LabelFrame(parent, text=title, bg=WINDOW_BG, fg=MUTED, padx=6, pady=6)
        canvas = tk.Canvas(frame, bg=PANEL_BG, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        return {"frame": frame, "label": canvas}

    def _get_threshold(self) -> float:
        try:
            value = float(self.threshold_var.get().strip().replace(",", "."))
        except Exception:
            value = DEFAULT_THRESHOLD
        return max(0.0, min(1.0, value))

    def load_a(self) -> None:
        self.path_a = self._pick_image()
        if self.path_a:
            self.status_var.set(f"Immagine A caricata: {os.path.basename(self.path_a)}")
            self.faces_a = []
            self.matches = []
            self.current_match_idx = -1

    def load_b(self) -> None:
        self.path_b = self._pick_image()
        if self.path_b:
            self.status_var.set(f"Immagine B caricata: {os.path.basename(self.path_b)}")
            self.faces_b = []
            self.matches = []
            self.current_match_idx = -1

    def _pick_image(self) -> Optional[str]:
        return filedialog.askopenfilename(
            title="Seleziona immagine",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"), ("All files", "*.*")],
        )

    def compare(self) -> None:
        try:
            if not self.path_a or not self.path_b:
                messagebox.showwarning("Attenzione", "Seleziona entrambe le immagini.")
                return

            self.faces_a = self._process_image_multi(self.path_a)
            self.faces_b = self._process_image_multi(self.path_b)

            if not self.faces_a or not self.faces_b:
                messagebox.showwarning("Attenzione", "Nessun volto utile trovato in una o entrambe le immagini.")
                return

            self.matches = []
            threshold = self._get_threshold()
            for ia, fa in enumerate(self.faces_a):
                for ib, fb in enumerate(self.faces_b):
                    if fa.embedding is None or fb.embedding is None:
                        continue
                    score = cosine_similarity(fa.embedding, fb.embedding)
                    verdict = self._interpret_similarity(score)
                    self.matches.append(MatchRecord(idx_a=ia, idx_b=ib, score=score, verdict=verdict))
            self.matches.sort(key=lambda m: m.score, reverse=True)

            self.current_match_idx = 0 if self.matches else -1
            if self.matches:
                self._show_match(0)
                self.status_var.set(
                    f"Confronto completato. Volti A: {len(self.faces_a)} | Volti B: {len(self.faces_b)} | Match: {len(self.matches)} | soglia {threshold:.2f}"
                )
            else:
                self._set_metrics("Nessun match disponibile.\n")
                self.nav_label.configure(text="Nessun match")
                self._draw_score_bar(None)

        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Errore", f"Errore durante il confronto:\n{exc}")

    def _process_image_multi(self, path: str) -> list[DetectedFace]:
        bgr = cv2.imread(path)
        if bgr is None:
            raise RuntimeError(f"Impossibile leggere il file: {path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_result = self.mp_face_mesh.process(rgb)
        mp_faces = []
        if mp_result.multi_face_landmarks:
            h, w = bgr.shape[:2]
            for face_landmarks in mp_result.multi_face_landmarks:
                pts = []
                xs, ys = [], []
                for lm in face_landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    pts.append((x, y))
                    xs.append(x)
                    ys.append(y)
                if xs and ys:
                    mp_faces.append({
                        "points": pts,
                        "bbox": (max(min(xs) - 20, 0), max(min(ys) - 20, 0), min(max(xs) + 20, w), min(max(ys) + 20, h)),
                    })

        arc_faces = self.arcface.get(bgr)
        total_faces_arcface = len(arc_faces)
        md5 = file_md5(path)
        sha256 = file_sha256(path)
        results: list[DetectedFace] = []

        if total_faces_arcface == 0:
            return results

        for idx, face in enumerate(sorted(arc_faces, key=lambda f: area_from_bbox(f.bbox), reverse=True), start=1):
            annotated = bgr.copy()
            emb = getattr(face, "embedding", None)
            embedding = np.asarray(emb, dtype=np.float32) if emb is not None else None
            similarity_ready = embedding is not None

            abox = face.bbox.astype(int)
            x1, y1, x2, y2 = map(int, abox)
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, bgr.shape[1])
            y2 = min(y2, bgr.shape[0])
            bbox = (x1, y1, x2, y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), YELLOW_BGR, 2)

            matched_mp = self._match_mp_face_to_bbox(mp_faces, bbox)
            mp_landmark_count = 0
            if matched_mp is not None:
                for x, y in matched_mp["points"]:
                    cv2.circle(annotated, (x, y), 1, YELLOW_BGR, -1, lineType=cv2.LINE_AA)
                mp_landmark_count = len(matched_mp["points"])

            crop = bgr[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else None
            warning = ""
            if matched_mp is None:
                warning = "Nessun set di landmark MediaPipe associato a questo volto."
            if not similarity_ready:
                warning = join_notes(warning, "Embedding ArcFace non disponibile.")
            if crop is None:
                warning = join_notes(warning, "Crop volto non disponibile.")

            results.append(
                DetectedFace(
                    image_path=path,
                    face_index=idx,
                    original_bgr=bgr,
                    annotated_bgr=annotated,
                    crop_bgr=crop,
                    embedding=embedding,
                    similarity_ready=similarity_ready,
                    total_faces_arcface=total_faces_arcface,
                    mp_landmark_count=mp_landmark_count,
                    bbox=bbox,
                    warning=warning,
                    md5=md5,
                    sha256=sha256,
                )
            )

        return results

    def _match_mp_face_to_bbox(self, mp_faces: list[dict], bbox: Tuple[int, int, int, int]) -> Optional[dict]:
        if not mp_faces:
            return None
        bx1, by1, bx2, by2 = bbox
        best = None
        best_score = -1.0
        for item in mp_faces:
            mx1, my1, mx2, my2 = item["bbox"]
            inter_x1 = max(bx1, mx1)
            inter_y1 = max(by1, my1)
            inter_x2 = min(bx2, mx2)
            inter_y2 = min(by2, my2)
            inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            area_b = max(1, (bx2 - bx1) * (by2 - by1))
            score = inter / area_b
            if score > best_score:
                best_score = score
                best = item
        return best if best_score > 0 else None

    def _interpret_similarity(self, score: float) -> str:
        threshold = self._get_threshold()
        medium = max(0.0, threshold - 0.20)
        if score >= threshold:
            return "Compatibilita alta"
        if score >= medium:
            return "Compatibilita media - valutare con cautela"
        return "Compatibilita bassa"

    def _show_match(self, idx: int) -> None:
        if not self.matches:
            return
        idx = max(0, min(idx, len(self.matches) - 1))
        self.current_match_idx = idx
        match = self.matches[idx]
        face_a = self.faces_a[match.idx_a]
        face_b = self.faces_b[match.idx_b]

        self._render_result(self.panel_a, face_a)
        self._render_result(self.panel_b, face_b)
        self._draw_score_bar(match.score)
        self.nav_label.configure(
            text=f"Match {idx + 1}/{len(self.matches)} | score {match.score:.4f} | volto A #{face_a.face_index}/{len(self.faces_a)} | volto B #{face_b.face_index}/{len(self.faces_b)}"
        )

        lines = []
        lines.append("=== RISULTATO COMPARAZIONE MULTI-VOLTO ===")
        lines.append(f"Somiglianza biometrica (ArcFace): {match.score:.4f}")
        lines.append(f"Soglia impostata: {self._get_threshold():.2f}")
        lines.append(f"Valutazione: {match.verdict}")
        lines.append("")
        lines.append(f"Immagine A: {face_a.image_path}")
        lines.append(f"Volto selezionato A: {face_a.face_index} su {len(self.faces_a)}")
        lines.append(f"MD5 A: {face_a.md5}")
        lines.append(f"SHA256 A: {face_a.sha256}")
        lines.append("")
        lines.append(f"Immagine B: {face_b.image_path}")
        lines.append(f"Volto selezionato B: {face_b.face_index} su {len(self.faces_b)}")
        lines.append(f"MD5 B: {face_b.md5}")
        lines.append(f"SHA256 B: {face_b.sha256}")
        lines.append("")
        lines.append("ArcFace viene usato per l'embedding e il confronto biometrico.")
        lines.append("MediaPipe viene usato per mostrare i keypoint gialli del volto selezionato.")
        self._set_metrics("\n".join(lines))

    def prev_match(self) -> None:
        if self.matches:
            self._show_match(self.current_match_idx - 1 if self.current_match_idx > 0 else 0)

    def next_match(self) -> None:
        if self.matches:
            self._show_match(self.current_match_idx + 1 if self.current_match_idx < len(self.matches) - 1 else self.current_match_idx)

    def _render_result(self, panel, result: DetectedFace) -> None:
        self._set_label_image(panel["original"], result.original_bgr, PREVIEW_SIZE)
        self._set_label_image(panel["annotated"], result.annotated_bgr, PREVIEW_SIZE)
        self._set_label_image(panel["crop"], result.crop_bgr, CROP_SIZE)
        info_lines = [
            f"File: {os.path.basename(result.image_path)}",
            f"Volto selezionato: {result.face_index} / {result.total_faces_arcface}",
            f"BBox: {result.bbox if result.bbox else 'Nessuna'}",
            f"Embedding ArcFace: {'Si' if result.similarity_ready else 'No'}",
            f"Landmark MediaPipe: {result.mp_landmark_count}",
            f"MD5: {result.md5}",
            f"SHA256: {result.sha256}",
            f"Avvisi: {result.warning or 'Nessuno'}",
        ]
        panel["info"].delete("1.0", tk.END)
        panel["info"].insert("1.0", "\n".join(info_lines))

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
    def _set_metrics(self, text: str) -> None:
        self.metrics.delete("1.0", tk.END)
        self.metrics.insert("1.0", text)

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
                left, top - 12,
                text=f"score corrente: {similarity:.4f} | range teorico -1.00 .. +1.00 | {verdict}",
                anchor="w", fill=TEXT_FG, font=("Helvetica", 10, "bold")
            )
        else:
            canvas.create_text(
                left, top - 12,
                text="Nessun punteggio disponibile. Il confronto usa ArcFace embedding + cosine similarity.",
                anchor="w", fill=TEXT_FG, font=("Helvetica", 10, "bold")
            )

    def export_csv(self) -> None:
        try:
            if not self.matches:
                messagebox.showwarning("Attenzione", "Esegui prima il confronto.")
                return
            save_path = filedialog.asksaveasfilename(
                title="Salva report CSV",
                defaultextension=".csv",
                initialfile=f"face_compare_multi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                filetypes=[("CSV", "*.csv")],
            )
            if not save_path:
                return

            with open(save_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow([
                    "rank", "score_arcface", "verdict",
                    "image_a", "face_idx_a", "total_faces_a", "md5_a", "sha256_a",
                    "image_b", "face_idx_b", "total_faces_b", "md5_b", "sha256_b"
                ])
                for i, m in enumerate(self.matches, start=1):
                    fa = self.faces_a[m.idx_a]
                    fb = self.faces_b[m.idx_b]
                    writer.writerow([
                        i, f"{m.score:.6f}", m.verdict,
                        fa.image_path, fa.face_index, fa.total_faces_arcface, fa.md5, fa.sha256,
                        fb.image_path, fb.face_index, fb.total_faces_arcface, fb.md5, fb.sha256
                    ])

            messagebox.showinfo("Completato", f"CSV salvato in:\n{save_path}")
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Errore", f"Impossibile esportare CSV:\n{exc}")

    def export_package(self) -> None:
        try:
            if not self.matches:
                messagebox.showwarning("Attenzione", "Esegui prima il confronto.")
                return

            base_dir = filedialog.askdirectory(title="Seleziona directory di destinazione")
            if not base_dir:
                return

            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_dir = Path(base_dir) / EXPORT_DIRNAME / f"compare_{ts}"
            out_dir.mkdir(parents=True, exist_ok=True)
            originals_dir = out_dir / "originals"
            annotated_dir = out_dir / "annotated"
            crops_dir = out_dir / "crops"
            originals_dir.mkdir(exist_ok=True)
            annotated_dir.mkdir(exist_ok=True)
            crops_dir.mkdir(exist_ok=True)

            if self.path_a:
                copy2(self.path_a, originals_dir / f"A_{Path(self.path_a).name}")
            if self.path_b:
                copy2(self.path_b, originals_dir / f"B_{Path(self.path_b).name}")

            for i, m in enumerate(self.matches[:100], start=1):
                fa = self.faces_a[m.idx_a]
                fb = self.faces_b[m.idx_b]
                cv2.imwrite(str(annotated_dir / f"{i:03d}_A_face{fa.face_index}.png"), fa.annotated_bgr)
                cv2.imwrite(str(annotated_dir / f"{i:03d}_B_face{fb.face_index}.png"), fb.annotated_bgr)
                if fa.crop_bgr is not None:
                    cv2.imwrite(str(crops_dir / f"{i:03d}_A_face{fa.face_index}.png"), fa.crop_bgr)
                if fb.crop_bgr is not None:
                    cv2.imwrite(str(crops_dir / f"{i:03d}_B_face{fb.face_index}.png"), fb.crop_bgr)

            (out_dir / "report.html").write_text(self._build_html(), encoding="utf-8")
            self.export_csv_to_path(out_dir / "report.csv")
            self._append_history_csv(out_dir)
            messagebox.showinfo("Completato", f"Pacchetto esportato in:\n{out_dir}")
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Errore", f"Impossibile esportare pacchetto:\n{exc}")

    def export_csv_to_path(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                "rank", "score_arcface", "verdict",
                "image_a", "face_idx_a", "total_faces_a", "md5_a", "sha256_a",
                "image_b", "face_idx_b", "total_faces_b", "md5_b", "sha256_b"
            ])
            for i, m in enumerate(self.matches, start=1):
                fa = self.faces_a[m.idx_a]
                fb = self.faces_b[m.idx_b]
                writer.writerow([
                    i, f"{m.score:.6f}", m.verdict,
                    fa.image_path, fa.face_index, fa.total_faces_arcface, fa.md5, fa.sha256,
                    fb.image_path, fb.face_index, fb.total_faces_arcface, fb.md5, fb.sha256
                ])

    def _append_history_csv(self, out_dir: Path) -> None:
        history_path = out_dir.parent / HISTORY_CSV
        header = ["timestamp", "image_a", "faces_a", "image_b", "faces_b", "matches", "threshold", "export_dir"]
        row = [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            self.path_a or "", len(self.faces_a),
            self.path_b or "", len(self.faces_b),
            len(self.matches), f"{self._get_threshold():.2f}",
            str(out_dir),
        ]
        append_csv_row(history_path, header, row)

    def export_html(self) -> None:
        try:
            if not self.matches:
                messagebox.showwarning("Attenzione", "Esegui prima il confronto.")
                return
            save_path = filedialog.asksaveasfilename(
                title="Salva report HTML",
                defaultextension=".html",
                initialfile=f"face_compare_multi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                filetypes=[("HTML", "*.html")],
            )
            if not save_path:
                return
            Path(save_path).write_text(self._build_html(), encoding="utf-8")
            messagebox.showinfo("Completato", f"Report salvato in:\n{save_path}")
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Errore", f"Impossibile esportare HTML:\n{exc}")

    def _build_html(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cards = []
        for i, m in enumerate(self.matches[:300], start=1):
            fa = self.faces_a[m.idx_a]
            fb = self.faces_b[m.idx_b]
            a_orig = img_to_b64(fa.original_bgr)
            a_ann = img_to_b64(fa.annotated_bgr)
            a_crop = img_to_b64(fa.crop_bgr)
            b_orig = img_to_b64(fb.original_bgr)
            b_ann = img_to_b64(fb.annotated_bgr)
            b_crop = img_to_b64(fb.crop_bgr)
            cards.append(f"""
<div class='card'>
<h2>Match {i} | Score {m.score:.4f} | {m.verdict}</h2>
<div class='grid3'>
<div><h3>A originale</h3><img src='data:image/png;base64,{a_orig}'></div>
<div><h3>A volto {fa.face_index}</h3><img src='data:image/png;base64,{a_ann}'></div>
<div><h3>A crop</h3><img src='data:image/png;base64,{a_crop}'></div>
</div>
<div class='grid3'>
<div><h3>B originale</h3><img src='data:image/png;base64,{b_orig}'></div>
<div><h3>B volto {fb.face_index}</h3><img src='data:image/png;base64,{b_ann}'></div>
<div><h3>B crop</h3><img src='data:image/png;base64,{b_crop}'></div>
</div>
<pre>Immagine A: {fa.image_path}
Volto A: {fa.face_index} / {fa.total_faces_arcface}
MD5 A: {fa.md5}
SHA256 A: {fa.sha256}

Immagine B: {fb.image_path}
Volto B: {fb.face_index} / {fb.total_faces_arcface}
MD5 B: {fb.md5}
SHA256 B: {fb.sha256}</pre>
</div>
""")
        return f"""<!DOCTYPE html>
<html lang='it'>
<head>
<meta charset='utf-8'>
<title>Face Compare Multi Report</title>
<style>
body {{ background:#111; color:#f0f0f0; font-family:Arial,Helvetica,sans-serif; margin:20px; }}
.card {{ background:#1b1b1b; border:1px solid #333; border-radius:12px; padding:16px; margin-bottom:20px; }}
.grid3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-bottom:12px; }}
img {{ max-width:100%; border-radius:8px; border:1px solid #444; }}
.h {{ color:#ffd400; }}
pre {{ white-space:pre-wrap; background:#161616; padding:12px; border-radius:8px; }}
</style>
</head>
<body>
<h1 class='h'>Face Compare Multi-Face - MediaPipe + ArcFace</h1>
<p>Generato il {now}</p>
<div class='card'>
<h2>Spiegazione tecnica</h2>
<pre>Il programma confronta tutti i volti rilevati nell'immagine A con tutti i volti rilevati nell'immagine B.

ArcFace / InsightFace viene usato per estrarre gli embedding facciali.
La similarita finale e calcolata con cosine similarity tra embedding.
MediaPipe viene usato per mostrare i keypoint gialli del volto selezionato.

Range teorico cosine similarity:
- minimo teorico: -1.00
- massimo teorico: +1.00

Soglia impostata in GUI:
- minimo: 0.00
- massimo: 1.00
- corrente: {self._get_threshold():.2f}</pre>
</div>
<div class='card'>
<pre>Immagine A: {self.path_a}
Volti rilevati A: {len(self.faces_a)}

Immagine B: {self.path_b}
Volti rilevati B: {len(self.faces_b)}

Match totali: {len(self.matches)}</pre>
</div>
{''.join(cards)}
</body>
</html>
"""


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
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
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


def img_to_b64(bgr: Optional[np.ndarray]) -> str:
    if bgr is None:
        blank = np.zeros((220, 220, 3), dtype=np.uint8)
        cv2.putText(blank, "N/D", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        bgr = blank
    ok, buf = cv2.imencode('.png', bgr)
    if not ok:
        raise RuntimeError('Impossibile convertire immagine in PNG')
    return base64.b64encode(buf.tobytes()).decode('ascii')


def main() -> None:
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except Exception:
        pass
    FaceCompareMultiApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
