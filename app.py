"""SignSpeak — train your own signs on webcam and translate them live.

New features:
1. Bidirectional: Sign → English + English → Text display
2. Motion sign mode (moving signs like ASL "cat")
3. Improved TTS (louder, clearer, small delay)
4. One-hand / Two-hand mode
5. Custom UI colors, font size, optional background image
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from camera import HandCamera
from features import POSE_SIZE, sequence_to_motion_features
from trainer import SignTrainer

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Default theme (can be overridden in Settings)
DEFAULTS = {
    "BG": "#0b1020",
    "CARD": "#141a2e",
    "ACCENT": "#6ee7d0",
    "ACCENT_DIM": "#2dd4bf",
    "TEXT": "#e8eefc",
    "MUTED": "#8b97b8",
    "DANGER": "#fb7185",
    "FONT_FAMILY": "Segoe UI",
    "FONT_SIZE": 14,
}

VIDEO_W, VIDEO_H = 720, 540
MOTION_BUFFER_LEN = 24  # ~0.8 s at 30 fps


class SignSpeakApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("SignSpeak")
        self.geometry("1320x800")
        self.minsize(1100, 700)

        self.theme = dict(DEFAULTS)
        self.configure(fg_color=self.theme["BG"])

        self.trainer = SignTrainer()
        self.latest_features: np.ndarray | None = None
        self.capturing = False
        self.capture_buffer: list[np.ndarray] = []
        self.photo = None
        self.last_committed: str | None = None
        self.last_commit_at: float = 0.0
        self.hold_started: float | None = None
        self.sentence_parts: list[str] = []
        self.pred_smooth: deque[tuple[str, float]] = deque(maxlen=6)
        self._tts_lock = threading.Lock()

        # New state
        self.one_hand_mode = False
        self.motion_mode = False
        self.motion_live_buffer: deque[np.ndarray] = deque(maxlen=MOTION_BUFFER_LEN)
        self.bg_image_path: str | None = None
        self._bg_photo = None

        self._build()
        self.camera = HandCamera(self._on_frame)
        self.camera.start()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(80, self._tick)

    # ───────────────────────── UI BUILD ─────────────────────────

    def _build(self) -> None:
        for w in self.winfo_children():
            w.destroy()

        t = self.theme
        root = ctk.CTkFrame(self, fg_color=t["BG"])
        root.pack(fill="both", expand=True, padx=18, pady=16)
        root.grid_columnconfigure(0, weight=3)
        root.grid_columnconfigure(1, weight=2)
        root.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(root, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        ctk.CTkLabel(
            header,
            text="SignSpeak",
            font=ctk.CTkFont(family=t["FONT_FAMILY"], size=28, weight="bold"),
            text_color=t["ACCENT"],
        ).pack(side="left")
        ctk.CTkLabel(
            header,
            text="  train your signs  ·  translate them live",
            font=ctk.CTkFont(family=t["FONT_FAMILY"], size=14),
            text_color=t["MUTED"],
        ).pack(side="left", pady=(8, 0))

        # Mode indicators
        mode_frame = ctk.CTkFrame(header, fg_color="transparent")
        mode_frame.pack(side="right")
        self.mode_label = ctk.CTkLabel(
            mode_frame,
            text=self._mode_text(),
            font=ctk.CTkFont(family=t["FONT_FAMILY"], size=12),
            text_color=t["MUTED"],
        )
        self.mode_label.pack(side="right")

        # Video card
        video_card = ctk.CTkFrame(root, fg_color=t["CARD"], corner_radius=18)
        video_card.grid(row=1, column=0, sticky="nsew", padx=(0, 14))
        self.video = ctk.CTkLabel(video_card, text="Starting camera…", text_color=t["MUTED"])
        self.video.pack(fill="both", expand=True, padx=12, pady=12)

        overlay = ctk.CTkFrame(video_card, fg_color="#0b1020", corner_radius=12)
        overlay.pack(fill="x", padx=16, pady=(0, 16))
        self.live_sign = ctk.CTkLabel(
            overlay,
            text="show a trained sign",
            font=ctk.CTkFont(family=t["FONT_FAMILY"], size=26, weight="bold"),
            text_color=t["TEXT"],
        )
        self.live_sign.pack(anchor="w", padx=16, pady=(12, 0))
        self.conf_bar = ctk.CTkProgressBar(overlay, height=8, progress_color=t["ACCENT_DIM"])
        self.conf_bar.pack(fill="x", padx=16, pady=(10, 6))
        self.conf_bar.set(0)
        self.conf_label = ctk.CTkLabel(overlay, text="confidence  0%", text_color=t["MUTED"])
        self.conf_label.pack(anchor="w", padx=16, pady=(0, 12))

        # Side panel
        side = ctk.CTkFrame(root, fg_color="transparent")
        side.grid(row=1, column=1, sticky="nsew")

        # Translation card
        sentence_card = ctk.CTkFrame(side, fg_color=t["CARD"], corner_radius=18)
        sentence_card.pack(fill="x")
        ctk.CTkLabel(
            sentence_card,
            text="TRANSLATION (Sign → English)",
            text_color=t["MUTED"],
            font=ctk.CTkFont(size=11, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(12, 0))
        self.sentence = ctk.CTkLabel(
            sentence_card,
            text="Your sentence will appear here.",
            wraplength=420,
            justify="left",
            font=ctk.CTkFont(family=t["FONT_FAMILY"], size=18),
            text_color=t["TEXT"],
        )
        self.sentence.pack(anchor="w", padx=16, pady=10)

        actions = ctk.CTkFrame(sentence_card, fg_color="transparent")
        actions.pack(fill="x", padx=12, pady=(0, 12))
        for txt, cmd in [
            ("Space", self._space),
            ("Delete", self._backspace),
            ("Clear", self._clear),
        ]:
            ctk.CTkButton(
                actions, text=txt, width=80, command=cmd,
                fg_color="#243049", hover_color="#31405d",
            ).pack(side="left", padx=4)
        ctk.CTkButton(
            actions, text="Speak", width=80, command=self._speak,
            fg_color=t["ACCENT_DIM"], text_color="#042f2e",
        ).pack(side="left", padx=4)

        # Tabs
        self.tabs = ctk.CTkTabview(
            side, fg_color=t["CARD"], corner_radius=18,
            segmented_button_selected_color=t["ACCENT_DIM"],
        )
        self.tabs.pack(fill="both", expand=True, pady=(14, 0))
        self.tabs.add("Train")
        self.tabs.add("English → Text")
        self.tabs.add("Library")
        self.tabs.add("Settings")
        self.tabs.add("How to")

        self._build_train_tab()
        self._build_english_tab()
        self._build_library_tab()
        self._build_settings_tab()
        self._build_howto_tab()

    def _mode_text(self) -> str:
        hand = "1-Hand" if self.one_hand_mode else "2-Hand"
        sign = "Motion" if self.motion_mode else "Static"
        return f"{hand}  ·  {sign}"

    def _build_train_tab(self) -> None:
        t = self.theme
        train = self.tabs.tab("Train")

        ctk.CTkLabel(
            train,
            text="Hold a sign in frame, name it, then capture samples.\n"
                 "Use Motion mode for signs that move (e.g. ASL cat).\n"
                 "Use 2-Hand mode when the sign needs both hands.",
            wraplength=400, justify="left", text_color=t["MUTED"],
        ).pack(anchor="w", padx=8, pady=(8, 10))

        # Mode toggles
        toggles = ctk.CTkFrame(train, fg_color="transparent")
        toggles.pack(fill="x", padx=8, pady=(0, 8))

        self.hand_switch = ctk.CTkSegmentedButton(
            toggles,
            values=["1 Hand", "2 Hands"],
            command=self._on_hand_mode,
            selected_color=t["ACCENT_DIM"],
        )
        self.hand_switch.set("2 Hands")
        self.hand_switch.pack(side="left", padx=(0, 12))

        self.motion_switch = ctk.CTkSegmentedButton(
            toggles,
            values=["Static", "Motion"],
            command=self._on_motion_mode,
            selected_color=t["ACCENT_DIM"],
        )
        self.motion_switch.set("Static")
        self.motion_switch.pack(side="left")

        self.sign_name = ctk.CTkEntry(
            train, placeholder_text="sign meaning  e.g. hello / cat", height=40,
        )
        self.sign_name.pack(fill="x", padx=8)

        self.capture_btn = ctk.CTkButton(
            train,
            text="Hold to capture  (2s)",
            height=44,
            fg_color=t["ACCENT_DIM"],
            hover_color=t["ACCENT"],
            text_color="#042f2e",
            font=ctk.CTkFont(weight="bold"),
        )
        self.capture_btn.pack(fill="x", padx=8, pady=(12, 6))
        self.capture_btn.bind("<ButtonPress-1>", self._start_capture)
        self.capture_btn.bind("<ButtonRelease-1>", self._stop_capture)

        self.capture_status = ctk.CTkLabel(train, text="No samples this session.", text_color=t["MUTED"])
        self.capture_status.pack(anchor="w", padx=8)

        self.train_btn = ctk.CTkButton(train, text="Train model", height=42, command=self._train_model)
        self.train_btn.pack(fill="x", padx=8, pady=(16, 6))
        self.train_status = ctk.CTkLabel(
            train, text=self._model_blurb(), text_color=t["MUTED"],
            wraplength=400, justify="left",
        )
        self.train_status.pack(anchor="w", padx=8)

    def _build_english_tab(self) -> None:
        """English → Text (display + speak). User asked for simple English text
        because they can read it."""
        t = self.theme
        tab = self.tabs.tab("English → Text")

        ctk.CTkLabel(
            tab,
            text="Type English text. It will be shown large so it can be read,\n"
                 "and you can speak it out loud.",
            wraplength=400, justify="left", text_color=t["MUTED"],
        ).pack(anchor="w", padx=8, pady=(8, 10))

        self.en_entry = ctk.CTkEntry(tab, placeholder_text="Type English here…", height=40)
        self.en_entry.pack(fill="x", padx=8)
        self.en_entry.bind("<Return>", lambda e: self._show_english())

        btn_row = ctk.CTkFrame(tab, fg_color="transparent")
        btn_row.pack(fill="x", padx=8, pady=(10, 6))
        ctk.CTkButton(
            btn_row, text="Show", width=100, command=self._show_english,
            fg_color=t["ACCENT_DIM"], text_color="#042f2e",
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            btn_row, text="Speak it", width=100, command=self._speak_english,
            fg_color="#243049", hover_color="#31405d",
        ).pack(side="left")

        self.en_display = ctk.CTkLabel(
            tab,
            text="English text will appear here large.",
            wraplength=400, justify="left",
            font=ctk.CTkFont(family=t["FONT_FAMILY"], size=22, weight="bold"),
            text_color=t["TEXT"],
        )
        self.en_display.pack(anchor="w", padx=8, pady=20)

    def _build_library_tab(self) -> None:
        library = self.tabs.tab("Library")
        self.lib_box = ctk.CTkScrollableFrame(library, fg_color="transparent")
        self.lib_box.pack(fill="both", expand=True, padx=4, pady=8)
        self._refresh_library()

    def _build_settings_tab(self) -> None:
        t = self.theme
        tab = self.tabs.tab("Settings")

        ctk.CTkLabel(
            tab,
            text="Customize colors (hex), font size, and optional background image.\n"
                 "Background image applies behind the main window (text stays readable).",
            wraplength=400, justify="left", text_color=t["MUTED"],
        ).pack(anchor="w", padx=8, pady=(8, 12))

        self.color_vars = {}
        for key, label in [
            ("BG", "Background"),
            ("CARD", "Card"),
            ("ACCENT", "Accent"),
            ("ACCENT_DIM", "Accent dim"),
            ("TEXT", "Text"),
            ("MUTED", "Muted text"),
        ]:
            row = ctk.CTkFrame(tab, fg_color="transparent")
            row.pack(fill="x", padx=8, pady=3)
            ctk.CTkLabel(row, text=label, width=100, anchor="w").pack(side="left")
            var = ctk.StringVar(value=t[key])
            entry = ctk.CTkEntry(row, textvariable=var, width=120)
            entry.pack(side="left")
            self.color_vars[key] = var

        # Font size
        row = ctk.CTkFrame(tab, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=6)
        ctk.CTkLabel(row, text="Font size", width=100, anchor="w").pack(side="left")
        self.font_size_var = ctk.StringVar(value=str(t["FONT_SIZE"]))
        ctk.CTkEntry(row, textvariable=self.font_size_var, width=60).pack(side="left")

        # Background image
        row = ctk.CTkFrame(tab, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=6)
        ctk.CTkLabel(row, text="BG image", width=100, anchor="w").pack(side="left")
        self.bg_path_var = ctk.StringVar(value=self.bg_image_path or "")
        ctk.CTkEntry(row, textvariable=self.bg_path_var, width=220,
                     placeholder_text="path to .png / .jpg (optional)").pack(side="left")

        btn_row = ctk.CTkFrame(tab, fg_color="transparent")
        btn_row.pack(fill="x", padx=8, pady=16)
        ctk.CTkButton(
            btn_row, text="Apply theme", command=self._apply_theme,
            fg_color=t["ACCENT_DIM"], text_color="#042f2e",
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            btn_row, text="Reset defaults", command=self._reset_theme,
            fg_color="#243049", hover_color="#31405d",
        ).pack(side="left")

    def _build_howto_tab(self) -> None:
        t = self.theme
        howto = self.tabs.tab("How to")
        tips = (
            "1. Open Train, pick 1-Hand or 2-Hands, Static or Motion.\n"
            "2. Type what the sign means (hello, cat, thanks…).\n"
            "3. Hold the pose (or move for Motion) and press Hold to capture.\n"
            "4. Capture several samples per sign, then hit Train model.\n"
            "5. Sign live — hold ~0.7 s (static) or perform the motion.\n"
            "6. Use Space / Delete to build a sentence, Speak to hear it.\n\n"
            "English → Text tab: type English, Show it large, or Speak it.\n\n"
            "Motion tips: perform the full movement while holding the button.\n"
            "Two-hand signs: switch to 2 Hands before capturing.\n\n"
            "Settings: change colors (hex codes), font size, or set a\n"
            "background image path (text stays on top)."
        )
        ctk.CTkLabel(howto, text=tips, justify="left", wraplength=400, text_color=t["TEXT"]).pack(
            anchor="w", padx=8, pady=12
        )

    # ───────────────────────── MODE HANDLERS ─────────────────────────

    def _on_hand_mode(self, value: str) -> None:
        self.one_hand_mode = value == "1 Hand"
        self.camera.set_one_hand_mode(self.one_hand_mode)
        self.mode_label.configure(text=self._mode_text())

    def _on_motion_mode(self, value: str) -> None:
        self.motion_mode = value == "Motion"
        self.motion_live_buffer.clear()
        self.mode_label.configure(text=self._mode_text())
        if self.motion_mode:
            self.capture_btn.configure(text="Hold to capture motion  (~1.5s)")
        else:
            self.capture_btn.configure(text="Hold to capture  (2s)")

    # ───────────────────────── CORE LOGIC ─────────────────────────

    def _model_blurb(self) -> str:
        n = len(self.trainer.labels())
        samples = len(self.trainer.y)
        static_ready = "Static ready" if self.trainer.model is not None else "No static model"
        motion_ready = "Motion ready" if self.trainer.motion_model is not None else "No motion model"
        return f"{static_ready}  ·  {motion_ready}  ·  {n} signs, {samples} samples."

    def _on_frame(self, frame_bgr: np.ndarray, features: np.ndarray | None) -> None:
        self.latest_features = features
        if self.capturing and features is not None:
            self.capture_buffer.append(features.copy())

        if features is not None and self.motion_mode and not self.capturing:
            self.motion_live_buffer.append(features.copy())

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize((VIDEO_W, VIDEO_H), Image.Resampling.BILINEAR)
        self.photo = ctk.CTkImage(light_image=image, dark_image=image, size=(VIDEO_W, VIDEO_H))
        self.after(0, lambda img=self.photo: self.video.configure(image=img, text=""))

    def _tick(self) -> None:
        if not self.capturing:
            if self.motion_mode:
                feat = sequence_to_motion_features(list(self.motion_live_buffer))
                label, conf = self.trainer.predict(feat, motion=True)
            else:
                label, conf = self.trainer.predict(self.latest_features, motion=False)

            if label:
                self.pred_smooth.append((label, conf))
            else:
                self.pred_smooth.clear()
                self.hold_started = None
                self.live_sign.configure(text="show a trained sign")
                self.conf_bar.set(0)
                self.conf_label.configure(text="confidence  0%")

            if self.pred_smooth:
                best = max(self.pred_smooth, key=lambda item: item[1])
                name, conf = best
                self.live_sign.configure(text=name)
                self.conf_bar.set(conf)
                self.conf_label.configure(text=f"confidence  {conf * 100:.0f}%")
                hold_time = 0.9 if self.motion_mode else 0.75
                if conf >= 0.58:
                    if self.hold_started is None:
                        self.hold_started = time.time()
                    elif time.time() - self.hold_started >= hold_time:
                        self._commit(name)
                        self.hold_started = None
                        if self.motion_mode:
                            self.motion_live_buffer.clear()
                else:
                    self.hold_started = None

        if self.capturing:
            n = len(self.capture_buffer)
            mode = "motion" if self.motion_mode else "pose"
            self.capture_status.configure(
                text=f"Capturing {mode}… {n} frames  (keep going)"
            )

        self.after(80, self._tick)

    def _commit(self, name: str) -> None:
        now = time.time()
        if name == self.last_committed and now - self.last_commit_at < 1.5:
            return
        self.last_committed = name
        self.last_commit_at = now
        self.sentence_parts.append(name)
        self._render_sentence()
        self.after(1500, self._unlock_commit)

    def _unlock_commit(self) -> None:
        self.last_committed = None

    def _render_sentence(self) -> None:
        text = " ".join(self.sentence_parts).strip() or "Your sentence will appear here."
        self.sentence.configure(text=text)

    def _space(self) -> None:
        if self.sentence_parts and self.sentence_parts[-1] != "":
            self.sentence_parts.append("")
        self._render_sentence()

    def _backspace(self) -> None:
        if self.sentence_parts:
            self.sentence_parts.pop()
        self._render_sentence()

    def _clear(self) -> None:
        self.sentence_parts.clear()
        self._render_sentence()

    # ───────────────────────── TTS ─────────────────────────

    def _speak(self) -> None:
        text = " ".join(part for part in self.sentence_parts if part).strip()
        if text:
            self._tts(text)

    def _speak_english(self) -> None:
        text = self.en_entry.get().strip()
        if text:
            self._tts(text)

    def _tts(self, text: str) -> None:
        """Louder, clearer TTS with a short intentional delay."""
        def run() -> None:
            with self._tts_lock:
                try:
                    time.sleep(0.25)  # small delay so it feels intentional
                    import pyttsx3
                    engine = pyttsx3.init()
                    # Volume max, slightly slower for clarity
                    engine.setProperty("volume", 1.0)
                    rate = engine.getProperty("rate")
                    engine.setProperty("rate", max(120, int(rate * 0.85)))
                    # Prefer a clearer voice if available
                    voices = engine.getProperty("voices")
                    for v in voices:
                        name = (v.name or "").lower()
                        if any(k in name for k in ("zira", "david", "samantha", "karen", "daniel")):
                            engine.setProperty("voice", v.id)
                            break
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    pass

        threading.Thread(target=run, daemon=True).start()

    def _show_english(self) -> None:
        text = self.en_entry.get().strip()
        if text:
            self.en_display.configure(text=text)

    # ───────────────────────── CAPTURE / TRAIN ─────────────────────────

    def _start_capture(self, _event=None) -> None:
        name = self.sign_name.get().strip()
        if not name:
            self.capture_status.configure(text="Type a sign meaning first.")
            return
        self.capturing = True
        self.capture_buffer = []
        self.capture_status.configure(text="Capturing… hold / move the sign.")

    def _stop_capture(self, _event=None) -> None:
        if not self.capturing:
            return
        self.capturing = False
        name = self.sign_name.get().strip()
        added = self.trainer.add_samples(
            name, self.capture_buffer, motion=self.motion_mode
        )
        self.capture_buffer = []
        if added == 0:
            self.capture_status.configure(
                text="No usable frames. Try better light / longer hold."
            )
        else:
            mode = "motion" if self.motion_mode else "static"
            self.capture_status.configure(
                text=f"Saved {added} {mode} sample(s) for “{name.lower()}”."
            )
        self.train_status.configure(text=self._model_blurb())
        self._refresh_library()

    def _train_model(self) -> None:
        try:
            stats = self.trainer.train(motion=self.motion_mode)
        except ValueError as exc:
            self.train_status.configure(text=str(exc))
            return
        self.train_status.configure(
            text=f"Trained {stats['mode']} · {stats['signs']} signs / "
                 f"{stats['samples']} samples  ·  fit {stats['accuracy'] * 100:.0f}%"
        )

    def _refresh_library(self) -> None:
        for child in self.lib_box.winfo_children():
            child.destroy()
        counts = self.trainer.counts()
        if not counts:
            ctk.CTkLabel(
                self.lib_box, text="No signs yet. Capture some in the Train tab.",
                text_color=self.theme["MUTED"],
            ).pack(anchor="w", padx=4, pady=8)
            return
        for label, n in counts.items():
            row = ctk.CTkFrame(self.lib_box, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(
                row, text=f"{label}  ({n})",
                text_color=self.theme["TEXT"],
            ).pack(side="left", padx=4)
            ctk.CTkButton(
                row, text="Delete", width=70, height=26,
                fg_color="#3a2030", hover_color=self.theme["DANGER"],
                command=lambda l=label: self._delete_label(l),
            ).pack(side="right", padx=4)

    def _delete_label(self, label: str) -> None:
        self.trainer.delete_label(label)
        self.train_status.configure(text=self._model_blurb())
        self._refresh_library()

    # ───────────────────────── THEME ─────────────────────────

    def _apply_theme(self) -> None:
        for key, var in self.color_vars.items():
            val = var.get().strip()
            if val.startswith("#") and len(val) in (4, 7):
                self.theme[key] = val
        try:
            size = int(self.font_size_var.get())
            if 10 <= size <= 28:
                self.theme["FONT_SIZE"] = size
        except ValueError:
            pass
        path = self.bg_path_var.get().strip()
        self.bg_image_path = path if path and Path(path).is_file() else None
        self._build()  # rebuild UI with new colors
        # re-attach camera callback is already done; just refresh mode labels
        self.hand_switch.set("1 Hand" if self.one_hand_mode else "2 Hands")
        self.motion_switch.set("Motion" if self.motion_mode else "Static")
        self.mode_label.configure(text=self._mode_text())
        self.train_status.configure(text=self._model_blurb())
        self._refresh_library()

    def _reset_theme(self) -> None:
        self.theme = dict(DEFAULTS)
        self.bg_image_path = None
        self._build()
        self.hand_switch.set("1 Hand" if self.one_hand_mode else "2 Hands")
        self.motion_switch.set("Motion" if self.motion_mode else "Static")
        self.mode_label.configure(text=self._mode_text())
        self.train_status.configure(text=self._model_blurb())
        self._refresh_library()

    def _close(self) -> None:
        self.camera.stop()
        self.destroy()


if __name__ == "__main__":
    SignSpeakApp().mainloop()
