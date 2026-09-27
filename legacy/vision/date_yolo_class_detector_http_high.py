import json
import os
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageTk
from ultralytics import YOLO


APP_DIR = Path(__file__).resolve().parent
DEFAULT_SETTINGS = APP_DIR / "camera_settings.json"
DEFAULT_MODEL = APP_DIR / "yolo26n.pt"


class DateDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Date Class Detector")
        self.root.geometry("1220x780")
        self.root.minsize(980, 680)

        self.cap = None
        self.model = None
        self.running = False
        self.after_id = None
        self.photo = None

        self.settings = {}
        self.software_brightness = 0.0
        self.software_contrast = 1.0

        # Robot HTTP command state. A command is sent only once while a date
        # remains visible. The system is re-armed after a short no-detection gap.
        self.command_in_progress = False
        self.command_armed = True
        self.no_detection_since = None
        self.rearm_delay = 0.75

        self.model_path = tk.StringVar(
            value=str(DEFAULT_MODEL) if DEFAULT_MODEL.exists() else ""
        )
        self.settings_path = tk.StringVar(value=str(DEFAULT_SETTINGS))
        self.camera_index = tk.IntVar(value=0)
        self.confidence = tk.DoubleVar(value=0.50)
        self.robot_url = tk.StringVar(value="http://127.0.0.1:8000")
        self.send_commands = tk.BooleanVar(value=True)

        self.class_id_text = tk.StringVar(value="—")
        self.class_name_text = tk.StringVar(value="No detection")
        self.confidence_text = tk.StringVar(value="")
        self.processing_text = tk.StringVar(value="Brightness +0 | Contrast 1.00×")
        self.robot_status_text = tk.StringVar(value="Robot: ready")

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_ui(self):
        top = tk.Frame(self.root, padx=10, pady=8)
        top.pack(fill="x")

        tk.Label(top, text="YOLO model:").grid(row=0, column=0, sticky="w")
        tk.Entry(top, textvariable=self.model_path, width=60).grid(
            row=0, column=1, padx=5, sticky="ew"
        )
        tk.Button(top, text="Select Model", command=self.select_model).grid(
            row=0, column=2, padx=5
        )

        tk.Label(top, text="Camera settings:").grid(row=1, column=0, sticky="w")
        tk.Entry(top, textvariable=self.settings_path, width=60).grid(
            row=1, column=1, padx=5, sticky="ew"
        )
        tk.Button(top, text="Select JSON", command=self.select_settings).grid(
            row=1, column=2, padx=5
        )

        tk.Label(top, text="Robot URL:").grid(row=2, column=0, sticky="w")
        tk.Entry(top, textvariable=self.robot_url, width=60).grid(
            row=2, column=1, padx=5, sticky="ew"
        )
        tk.Checkbutton(
            top, text="Send commands", variable=self.send_commands
        ).grid(row=2, column=2, padx=5, sticky="w")

        controls = tk.Frame(top)
        controls.grid(row=3, column=0, columnspan=3, pady=(8, 0), sticky="w")

        tk.Label(controls, text="Camera:").pack(side="left")
        tk.Spinbox(
            controls, from_=0, to=10, width=4, textvariable=self.camera_index
        ).pack(side="left", padx=(4, 12))

        tk.Label(controls, text="Confidence:").pack(side="left")
        tk.Scale(
            controls,
            from_=0.05,
            to=0.95,
            resolution=0.05,
            orient="horizontal",
            variable=self.confidence,
            length=180,
        ).pack(side="left", padx=(4, 12))

        self.start_button = tk.Button(
            controls, text="Start", width=10, command=self.start
        )
        self.start_button.pack(side="left", padx=4)

        self.stop_button = tk.Button(
            controls, text="Stop", width=10, command=self.stop, state="disabled"
        )
        self.stop_button.pack(side="left", padx=4)

        self.reload_button = tk.Button(
            controls,
            text="Reload Settings",
            width=14,
            command=self.reload_settings,
            state="disabled",
        )
        self.reload_button.pack(side="left", padx=4)

        self.native_button = tk.Button(
            controls,
            text="Camera Properties",
            width=15,
            command=self.open_native_camera_settings,
            state="disabled",
        )
        self.native_button.pack(side="left", padx=4)

        top.grid_columnconfigure(1, weight=1)

        body = tk.Frame(self.root, padx=10, pady=10)
        body.pack(fill="both", expand=True)

        camera_frame = tk.LabelFrame(body, text="Processed Image + YOLO", padx=5, pady=5)
        camera_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self.video_label = tk.Label(
            camera_frame,
            text="Camera stopped",
            bg="black",
            fg="white",
            font=("Arial", 18),
        )
        self.video_label.pack(fill="both", expand=True)

        tk.Label(
            camera_frame,
            textvariable=self.processing_text,
            font=("Arial", 11),
        ).pack(fill="x", pady=(4, 0))

        result_frame = tk.LabelFrame(body, text="Detected Date", padx=10, pady=10)
        result_frame.pack(side="right", fill="both", expand=False)
        result_frame.configure(width=360)
        result_frame.pack_propagate(False)

        tk.Label(
            result_frame,
            text="CLASS ID",
            font=("Arial", 20, "bold"),
        ).pack(pady=(40, 0))

        self.big_class_label = tk.Label(
            result_frame,
            textvariable=self.class_id_text,
            font=("Arial", 150, "bold"),
            anchor="center",
        )
        self.big_class_label.pack(fill="x", pady=(0, 5))

        tk.Label(
            result_frame,
            textvariable=self.class_name_text,
            font=("Arial", 24, "bold"),
            wraplength=320,
            justify="center",
        ).pack(pady=5)

        tk.Label(
            result_frame,
            textvariable=self.confidence_text,
            font=("Arial", 18),
        ).pack(pady=5)

        tk.Label(
            result_frame,
            textvariable=self.robot_status_text,
            font=("Arial", 12),
            wraplength=320,
            justify="center",
        ).pack(pady=(20, 5))

        self.status = tk.StringVar(value="Ready")
        tk.Label(
            self.root,
            textvariable=self.status,
            relief="sunken",
            anchor="w",
            padx=8,
        ).pack(fill="x", side="bottom")

    def select_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO model",
            filetypes=[
                ("PyTorch model", "*.pt"),
                ("ONNX model", "*.onnx"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.model_path.set(path)

    def select_settings(self):
        path = filedialog.askopenfilename(
            title="Select camera settings JSON",
            filetypes=[("JSON file", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.settings_path.set(path)

    def load_settings_file(self):
        path = Path(self.settings_path.get()).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Camera settings file not found:\n{path}")

        with path.open("r", encoding="utf-8") as f:
            settings = json.load(f)

        print("\n" + "=" * 72)
        print(f"[Settings] Loading: {path}")
        print(json.dumps(settings, indent=2))
        return settings

    @staticmethod
    def bool_value(value):
        """
        JSON camera UIs often save 0/1 rather than true/false.
        """
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def configure_software_processing(self, settings):
        """
        Brightness and Contrast from the original tuning application are
        POST-PROCESSING controls, not hardware webcam properties.

        Brightness:
            additive offset, normally -100..100

        Contrast:
            multiplicative factor.
            If saved as 258, interpret it as 2.58x.
            If saved as 2.58, use it directly.
        """
        brightness = float(settings.get("Brightness", 0))
        contrast_raw = float(settings.get("Contrast", 1.0))

        if contrast_raw > 10:
            contrast = contrast_raw / 100.0
        else:
            contrast = contrast_raw

        self.software_brightness = brightness
        self.software_contrast = contrast

        self.processing_text.set(
            f"Software processing: Brightness {brightness:+.0f} | "
            f"Contrast {contrast:.2f}×"
        )

        print(
            f"[Processing] Brightness = {brightness:+.1f} "
            f"(software additive offset)"
        )
        print(
            f"[Processing] Contrast   = {contrast:.3f}× "
            f"(software multiplier; saved value={contrast_raw:g})"
        )

    def set_camera_property(self, label, prop, requested, pause=0.10):
        """
        Try to set one OpenCV camera property and verify by reading it back.

        Note: camera drivers may quantize, clamp, ignore or expose a different
        representation of a property. cap.set() returning True is not enough.
        """
        before = self.cap.get(prop)

        try:
            set_result = self.cap.set(prop, float(requested))
        except Exception as exc:
            print(f"[Camera] {label}: ERROR while setting: {exc}")
            return None

        time.sleep(pause)
        actual = self.cap.get(prop)

        if actual < 0:
            verification = "UNVERIFIED/UNSUPPORTED"
        elif abs(actual - float(requested)) < 0.01:
            verification = "EXACT"
        else:
            verification = "DIFFERENT"

        print(
            f"[Camera] {label}: before={before:g}, requested={requested}, "
            f"set_return={set_result}, actual={actual:g} -> {verification}"
        )
        return actual

    def apply_camera_settings(self, settings):
        if self.cap is None:
            return

        try:
            backend = self.cap.getBackendName()
        except Exception:
            backend = "unknown"

        print(f"[Camera] Backend: {backend}")

        # Let the camera/driver initialize before changing controls.
        for _ in range(5):
            self.cap.read()
        time.sleep(0.20)

        # ------------------------------------------------------------
        # AUTO EXPOSURE
        #
        # For OpenCV + DirectShow/V4L2, CAP_PROP_AUTO_EXPOSURE commonly
        # uses 0.25 = manual, 0.75 = auto rather than JSON boolean 0/1.
        # ------------------------------------------------------------
        auto_exposure = self.bool_value(settings.get("Auto Exposure", 0))
        ae_backend_value = 0.75 if auto_exposure else 0.25
        self.set_camera_property(
            "Auto Exposure",
            cv2.CAP_PROP_AUTO_EXPOSURE,
            ae_backend_value,
        )

        if auto_exposure:
            if "Exposure" in settings:
                print(
                    "[Camera] Exposure: skipped because Auto Exposure is enabled "
                    f"(saved manual value={settings['Exposure']})"
                )
        elif "Exposure" in settings:
            self.set_camera_property(
                "Exposure",
                cv2.CAP_PROP_EXPOSURE,
                settings["Exposure"],
            )

        # ------------------------------------------------------------
        # AUTO FOCUS / MANUAL FOCUS
        # ------------------------------------------------------------
        auto_focus = self.bool_value(settings.get("Auto Focus", 0))
        self.set_camera_property(
            "Auto Focus",
            cv2.CAP_PROP_AUTOFOCUS,
            1 if auto_focus else 0,
        )

        if auto_focus:
            if "Focus" in settings:
                print(
                    "[Camera] Focus: skipped because Auto Focus is enabled "
                    f"(saved manual value={settings['Focus']})"
                )
        elif "Focus" in settings:
            self.set_camera_property(
                "Focus",
                cv2.CAP_PROP_FOCUS,
                settings["Focus"],
                pause=0.20,
            )

        # ------------------------------------------------------------
        # AUTO WHITE BALANCE / MANUAL WB TEMPERATURE
        # ------------------------------------------------------------
        auto_wb = self.bool_value(settings.get("Auto White Balance", 0))
        self.set_camera_property(
            "Auto White Balance",
            cv2.CAP_PROP_AUTO_WB,
            1 if auto_wb else 0,
        )

        if auto_wb:
            if "WB Temperature" in settings:
                print(
                    "[Camera] WB Temperature: skipped because Auto White Balance "
                    f"is enabled (saved manual value={settings['WB Temperature']})"
                )
        elif "WB Temperature" in settings:
            self.set_camera_property(
                "WB Temperature",
                cv2.CAP_PROP_WB_TEMPERATURE,
                settings["WB Temperature"],
                pause=0.20,
            )

        # IMPORTANT:
        # Do NOT call CAP_PROP_BRIGHTNESS / CAP_PROP_CONTRAST here.
        # These two saved values belong to software post-processing.
        self.configure_software_processing(settings)

        print("=" * 72 + "\n")

    def process_image(self, frame):
        """
        Apply exactly the same style of digital correction used by the
        camera-tuning workflow, before YOLO inference.

        output = frame * contrast + brightness
        """
        processed = (
            frame.astype(np.float32) * self.software_contrast
            + self.software_brightness
        )
        return np.clip(processed, 0, 255).astype(np.uint8)

    def open_camera(self):
        index = int(self.camera_index.get())

        if os.name == "nt":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}.")

        return cap

    def start(self):
        if self.running:
            return

        model_path = self.model_path.get().strip()
        if not model_path:
            messagebox.showerror(
                "Model required",
                "Select a YOLO model (.pt or .onnx) before starting.",
            )
            return

        if not Path(model_path).exists():
            messagebox.showerror(
                "Model not found",
                f"The selected model does not exist:\n{model_path}",
            )
            return

        try:
            self.status.set("Loading YOLO model...")
            self.root.update_idletasks()
            self.model = YOLO(model_path)

            self.settings = self.load_settings_file()

            self.status.set("Opening camera...")
            self.root.update_idletasks()
            self.cap = self.open_camera()

            self.apply_camera_settings(self.settings)

            self.command_armed = True
            self.command_in_progress = False
            self.no_detection_since = None
            self.robot_status_text.set("Robot: ready")

            self.running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.reload_button.config(state="normal")
            self.native_button.config(state="normal")
            self.status.set("Detection running")
            self.update_frame()

        except Exception as exc:
            self.cleanup_camera()
            messagebox.showerror("Start error", str(exc))
            self.status.set("Error")

    def reload_settings(self):
        if self.cap is None:
            return

        try:
            self.settings = self.load_settings_file()
            self.apply_camera_settings(self.settings)
            self.status.set("Settings reloaded")
        except Exception as exc:
            messagebox.showerror("Settings error", str(exc))

    def open_native_camera_settings(self):
        """
        DirectShow only: ask the webcam driver to show its native property dialog.
        This is useful for seeing the device's real supported ranges/steps.
        """
        if self.cap is None:
            return

        try:
            ok = self.cap.set(cv2.CAP_PROP_SETTINGS, 1)
            print(f"[Camera] Native properties dialog requested: {ok}")
        except Exception as exc:
            messagebox.showerror(
                "Camera properties",
                f"Could not open the native camera properties dialog:\n{exc}",
            )

    def command_endpoint(self):
        """Return the configured /command URL."""
        base = self.robot_url.get().strip()
        if not base:
            return ""
        if base.rstrip("/").endswith("/command"):
            return base.rstrip("/")
        return base.rstrip("/") + "/command"

    def maybe_send_move_command(self, class_id):
        """
        Send one move command for the current detection event.

        Keeping the command armed/disarmed prevents a stationary date from
        generating dozens of POST requests every second.
        """
        if not self.send_commands.get():
            return
        if not self.command_armed or self.command_in_progress:
            return

        endpoint = self.command_endpoint()
        if not endpoint:
            self.robot_status_text.set("Robot: URL not configured")
            return

        payload = {
            "command": "move",
            "location": f"P{int(class_id)}",
            "height": "high",
        }

        self.command_armed = False
        self.command_in_progress = True
        self.robot_status_text.set(f"Robot: sending {payload['location']}...")

        print(f"[HTTP] POST {endpoint}")
        print(f"[HTTP] JSON {json.dumps(payload)}")

        thread = threading.Thread(
            target=self.send_move_command_worker,
            args=(endpoint, payload),
            daemon=True,
        )
        thread.start()

    def send_move_command_worker(self, endpoint, payload):
        """Run the HTTP request outside the Tkinter/video thread."""
        try:
            response = requests.post(endpoint, json=payload, timeout=5)
            response.raise_for_status()

            text = response.text.strip()
            if len(text) > 120:
                text = text[:117] + "..."

            print(
                f"[HTTP] Success: {response.status_code} "
                f"{text if text else '(empty response)'}"
            )
            self.root.after(
                0,
                lambda: self.robot_status_text.set(
                    f"Robot: {payload['location']} sent ({response.status_code})"
                ),
            )
        except requests.RequestException as exc:
            print(f"[HTTP] ERROR: {exc}")
            self.root.after(
                0,
                lambda e=str(exc): self.robot_status_text.set(f"Robot error: {e}"),
            )
        finally:
            self.root.after(0, self.finish_command_request)

    def finish_command_request(self):
        self.command_in_progress = False

    def update_detection_rearm(self, detected):
        """
        Re-arm command sending only after the camera has seen no date for
        rearm_delay seconds. This allows the same class to be processed again
        after one physical date is removed and another is presented.
        """
        if detected:
            self.no_detection_since = None
            return

        if self.no_detection_since is None:
            self.no_detection_since = time.monotonic()
            return

        if (
            not self.command_armed
            and not self.command_in_progress
            and time.monotonic() - self.no_detection_since >= self.rearm_delay
        ):
            self.command_armed = True
            self.robot_status_text.set("Robot: ready for next date")

    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.status.set("Could not read camera frame")
            self.after_id = self.root.after(100, self.update_frame)
            return

        try:
            # Digital brightness/contrast MUST happen before YOLO.
            processed = self.process_image(frame)

            results = self.model.predict(
                source=processed,
                conf=float(self.confidence.get()),
                verbose=False,
            )

            result = results[0]
            annotated = result.plot()

            if result.boxes is not None and len(result.boxes) > 0:
                confs = result.boxes.conf.cpu().tolist()
                classes = result.boxes.cls.cpu().tolist()

                # If several dates exist, display the highest-confidence class.
                best_index = max(range(len(confs)), key=confs.__getitem__)
                class_id = int(classes[best_index])
                conf = float(confs[best_index])

                class_name = str(class_id)
                if hasattr(result, "names") and result.names is not None:
                    class_name = result.names.get(class_id, str(class_id))

                self.class_id_text.set(str(class_id))
                self.class_name_text.set(class_name)
                self.confidence_text.set(f"Confidence: {conf:.1%}")

                self.update_detection_rearm(True)
                self.maybe_send_move_command(class_id)
            else:
                self.class_id_text.set("—")
                self.class_name_text.set("No detection")
                self.confidence_text.set("")
                self.update_detection_rearm(False)

            self.show_frame(annotated)

        except Exception as exc:
            self.status.set(f"Inference error: {exc}")
            self.show_frame(frame)

        self.after_id = self.root.after(10, self.update_frame)

    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        max_w = max(self.video_label.winfo_width(), 640)
        max_h = max(self.video_label.winfo_height(), 480)
        img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(img)
        self.video_label.config(image=self.photo, text="")

    def stop(self):
        self.running = False

        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

        self.cleanup_camera()

        self.video_label.config(image="", text="Camera stopped")
        self.photo = None

        self.class_id_text.set("—")
        self.class_name_text.set("No detection")
        self.confidence_text.set("")
        self.command_armed = True
        self.no_detection_since = None
        self.robot_status_text.set("Robot: stopped")

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.reload_button.config(state="disabled")
        self.native_button.config(state="disabled")
        self.status.set("Stopped")

    def cleanup_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def on_close(self):
        self.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = DateDetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
