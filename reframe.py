#!/usr/bin/env python3

import os
import sys
import json
import subprocess
import logging
import socket
from logging.handlers import RotatingFileHandler
from time import sleep

# Lazy-loaded by _lazy_import_pil() on first use
Image = None
ImageEnhance = None

def _lazy_import_pil():
    """Import PIL only when needed for image processing."""
    global Image, ImageEnhance
    if Image is None:
        from PIL import Image, ImageEnhance
    return Image, ImageEnhance

from picamera2 import Picamera2

from typing import Optional, Dict, Any

np = None

def _lazy_import_numpy():
    """Import NumPy only when image processing/display conversion needs it."""
    global np
    if np is None:
        import numpy as _np
        np = _np
    return np

# Lazy-loaded by _lazy_import_fastapi() on first use
_API_AVAILABLE = None
FastAPI = None
HTTPException = None
Request = None
uvicorn = None

def _lazy_import_fastapi():
    """Import FastAPI/uvicorn only when needed."""
    global _API_AVAILABLE, FastAPI, HTTPException, Request, uvicorn
    if _API_AVAILABLE is None:
        try:
            from fastapi import FastAPI, HTTPException, Request
            import uvicorn
            _API_AVAILABLE = True
        except Exception:
            _API_AVAILABLE = False
            FastAPI = None
            HTTPException = None
            Request = None
            uvicorn = None
    return _API_AVAILABLE

import threading
import time

# ═══════════════════════════════════════════════════════════════════
# HARDWARE: Display — defaults to Waveshare 4" ePaper Spectra 6
# The driver lives in waveshare_epd/. To use a different e-ink display,
# update the DISPLAY_* constants, ImageProcessor palette/buffer mapping, and
# the EInkDisplay adapter class below.
# ═══════════════════════════════════════════════════════════════════
libdir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib')
if os.path.exists(libdir):
    sys.path.append(libdir)

# Constants for file paths
BASE_PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_PATH = os.path.join(BASE_PATH, "photos")
PROCESSED_PATH = os.path.join(BASE_PATH, "dithered_photos")
RUNTIME_PATH = os.path.join(BASE_PATH, ".runtime")
DIAGNOSTICS_PATH = os.path.join(RUNTIME_PATH, "diagnostics.json")
LOG_PATH = os.path.join(RUNTIME_PATH, "reframe.log")
HDR_HELPER_PATH = os.path.join(BASE_PATH, "scripts", "enable_hdr.sh")
ORIGINAL_CAPTURE_EXTENSION = "jpg"
DISPLAY_IMAGE_WIDTH = 600
DISPLAY_IMAGE_HEIGHT = 400
DISPLAY_PANEL_WIDTH = 400
DISPLAY_PANEL_HEIGHT = 600
DISPLAY_IMAGE_SIZE = (DISPLAY_IMAGE_WIDTH, DISPLAY_IMAGE_HEIGHT)
DISPLAY_PANEL_SIZE = (DISPLAY_PANEL_WIDTH, DISPLAY_PANEL_HEIGHT)
BUTTON_POLL_INTERVAL_SECONDS = 0.025

os.makedirs(RUNTIME_PATH, exist_ok=True)
_log_handlers = [logging.StreamHandler()]
try:
    _log_handlers.append(RotatingFileHandler(LOG_PATH, maxBytes=100_000, backupCount=1))
except OSError:
    pass
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=_log_handlers,
    force=True,
)

_diagnostics_lock = threading.Lock()
_diagnostics = {"issues": {}, "last_activity": None}


def update_diagnostics(component=None, issue=None, activity=None):
    """Persist current issues and one useful recent activity."""
    with _diagnostics_lock:
        changed = False
        if component:
            if issue:
                _diagnostics["issues"][component] = issue
                changed = True
            else:
                changed = _diagnostics["issues"].pop(component, None) is not None
        if activity:
            _diagnostics["last_activity"] = activity
            changed = True

        if not changed:
            return

        temp_path = f"{DIAGNOSTICS_PATH}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as diagnostic_file:
                json.dump(_diagnostics, diagnostic_file)
            os.replace(temp_path, DIAGNOSTICS_PATH)
        except OSError as error:
            logging.warning("Could not persist diagnostics: %s", error)


def get_diagnostics():
    with _diagnostics_lock:
        return json.loads(json.dumps(_diagnostics))


def record_service_failure(error):
    """Classify common startup failures into useful physical checks."""
    lowered = str(error).lower()

    if any(marker in lowered for marker in (
        "no cameras available",
        "camera __init__ sequence did not complete",
        "failed to acquire camera",
        "pipeline handler",
        "libcamera",
    )):
        component = "camera"
        message = (
            "Camera not detected. Check that the ribbon cable is fully seated "
            "and facing the correct direction at both ends."
        )
    elif any(marker in lowered for marker in ("i2c", "smbus", "/dev/i2c")):
        component = "button"
        message = "Shutter button hardware is unavailable. Check the PiSugar connection and I2C."
    elif any(marker in lowered for marker in ("spi", "epd", "e-ink", "eink")):
        component = "display"
        message = "Display hardware failed to initialize. Check the display connection and SPI."
    else:
        component = "service"
        message = "Camera service failed to start."

    update_diagnostics(
        component,
        message,
        "Camera service stopped unexpectedly",
    )

# Color palettes for dithering. We blend between the two to create a saturated look while preserving details.
# Idea from https://github.com/pimoroni/inky
DESATURATED_PALETTE = [
    [0, 0, 0],          # Black
    [255, 255, 255],    # White
    [0, 255, 0],        # Green
    [0, 0, 255],        # Blue
    [255, 0, 0],        # Red
    [255, 255, 0],      # Yellow
]

SATURATED_PALETTE = [
    [57, 48, 57],       # Muted Black
    [255, 255, 255],    # White
    [40, 91, 58],       # Muted Green
    [0, 128, 255],      # Muted Blue
    [156, 72, 75],      # Muted Red
    [208, 190, 71],     # Muted Yellow
]

def get_dashboard_access_info(hostname=None, ip_address=None):
    """Build the friendly dashboard URLs for this device."""
    if hostname is None:
        hostname = socket.gethostname()
    hostname = (hostname or "reframe").split(".")[0]

    if ip_address is None:
        ip_address = get_lan_ip_address()

    primary_url = f"http://{hostname}.local"
    fallback_url = f"http://{ip_address}" if ip_address else None
    marker = f"{primary_url}|{fallback_url or ''}"
    return {
        "hostname": hostname,
        "ip_address": ip_address,
        "primary_url": primary_url,
        "fallback_url": fallback_url,
        "marker": marker
    }


def get_lan_ip_address():
    """Return wlan0's usable IPv4 address, if it has one."""
    try:
        result = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", "dev", "wlan0", "scope", "global"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            fields = result.stdout.split()
            for index, field in enumerate(fields[:-1]):
                if field != "inet":
                    continue
                ip_address = fields[index + 1].split("/", 1)[0]
                if _is_usable_lan_ip(ip_address):
                    return ip_address
    except Exception:
        pass

    return None


def _is_usable_lan_ip(ip_address):
    return (
        ip_address
        and "." in ip_address
        and not ip_address.startswith("127.")
        and not ip_address.startswith("169.254.")
    )


def _enable_camera_hdr():
    """Enable Camera Module 3 HDR after imports but before Picamera2 opens it."""
    if not os.path.isfile(HDR_HELPER_PATH):
        logging.warning("HDR helper not found at %s; continuing without HDR", HDR_HELPER_PATH)
        return

    try:
        result = subprocess.run([HDR_HELPER_PATH], timeout=4, check=False)
        if result.returncode != 0:
            logging.warning("HDR helper exited with status %s", result.returncode)
    except subprocess.TimeoutExpired:
        logging.warning("HDR helper timed out; continuing without HDR")
    except Exception as e:
        logging.warning("Could not run HDR helper: %s", e)


def _notify_systemd_ready(status):
    """Tell systemd startup capture dispatch is complete."""
    notify_socket = os.environ.get("NOTIFY_SOCKET")
    if not notify_socket:
        return

    address = notify_socket
    if address.startswith("@"):
        address = "\0" + address[1:]

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as notifier:
            notifier.connect(address)
            notifier.send(f"READY=1\nSTATUS={status}".encode("utf-8"))
        logging.info("Notified systemd: %s", status)
    except Exception as e:
        logging.error("Could not notify systemd that startup is ready: %s", e)


def _auto_display_enabled(settings_path):
    """Read only the startup display flag before CameraManager is constructed."""
    try:
        with open(settings_path, "r") as settings_file:
            settings = json.load(settings_file)
        display_settings = settings.get("display", {})
        if isinstance(display_settings, dict):
            return display_settings.get("auto_display", True)
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    return True


def render_dashboard_qr_image(access_info):
    """Render a dashboard QR screen for the ePaper display."""
    Image, _ = _lazy_import_pil()
    from PIL import ImageDraw
    import qrcode

    primary_url = access_info["primary_url"]
    fallback_url = access_info.get("fallback_url")

    qr_url = fallback_url or primary_url
    qr = qrcode.QRCode(border=2, box_size=8)
    qr.add_data(qr_url)
    qr.make(fit=True)
    qr_image = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    qr_image = qr_image.resize((250, 250))

    canvas = Image.new("RGB", DISPLAY_IMAGE_SIZE, "white")
    canvas.paste(qr_image, (30, 75))

    draw = ImageDraw.Draw(canvas)
    x = 315
    draw.text((x, 90), "reFrame dashboard", fill="black")
    draw.text((x, 125), primary_url, fill="black")
    if fallback_url:
        draw.text((x, 170), "If that does not open:", fill="black")
        draw.text((x, 200), fallback_url, fill="black")
    draw.text((x, 255), "Scan with your phone", fill="black")
    draw.text((x, 285), "to open photos.", fill="black")

    return canvas

class CameraManager:
    """Picamera2 camera adapter.

    To support a different camera stack, keep this public surface compatible:
    load/reload settings, configure/start the camera, capture a PIL RGB image,
    optionally save a file capture, and expose timeout/shutdown helpers used by
    CameraSystem. The rest of the app only expects capture_image_with_metadata()
    to return (result_dict, PIL_image).
    """

    def __init__(self, settings_path="settings.json"):
        self.settings_path = settings_path
        self.settings = self.load_settings()
        self.picam2 = Picamera2()
        self.last_activity_monotonic = time.monotonic()
        self._has_captured = False  # Track if we've taken at least one photo (for adaptive AF)
        self.configure_camera()

    def load_settings(self):
        """Load camera settings from JSON file."""
        try:
            with open(self.settings_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning(f"Could not load settings from {self.settings_path}: {e}")
            # Return default settings
            return {
                "camera": {
                    "resolution": {"width": 1200, "height": 800},
                    "exposure_value": 0,
                    "sharpness": 3,
                    "autofocus_mode": 2
                },
                "processing": {
                    "saturation": 0.6,
                    "brightness_factor": 1.1,
                    "color_factor": 1.4,
                    "dithering_method": "floyd_steinberg",
                    "bayer_size": 4,
                    "threshold_scale": 1.0
                },
                "display": {
                    "auto_display": True,
                    "display_timeout": 0,
                    "fast_refresh": False
                },
                "system": {
                    "auto_refresh_interval": 30,
                    "auto_timeout_minutes": 10,
                    "auto_timeout_enabled": True,
                    "show_dashboard_qr_on_wifi_connect": True
                }
            }

    def reload_settings(self):
        """Reload settings from file and reconfigure camera."""
        old_settings = self.settings.copy()
        self.settings = self.load_settings()

        # Only reconfigure if camera settings changed
        camera_changed = old_settings.get("camera", {}) != self.settings.get("camera", {})
        if camera_changed:
            logging.info("Camera settings changed, reconfiguring...")
            self.configure_camera()

        return self.settings

    def apply_camera_settings(self, camera_settings=None):
        """Apply specific camera settings without full reconfiguration."""
        if camera_settings is None:
            camera_settings = self.settings.get("camera", {})

        # Update only the controls that can be changed while running
        controls = {}

        if "exposure_value" in camera_settings:
            controls["ExposureValue"] = camera_settings["exposure_value"]
        if "sharpness" in camera_settings:
            controls["Sharpness"] = camera_settings["sharpness"]
        if "autofocus_mode" in camera_settings:
            controls["AfMode"] = camera_settings["autofocus_mode"]

        # Apply controls one by one to handle unsupported controls gracefully
        for control_name, control_value in controls.items():
            try:
                self.picam2.set_controls({control_name: control_value})
                logging.info(f"Applied {control_name}: {control_value}")
            except Exception as e:
                logging.warning(f"Could not set {control_name}: {e}")

    def capture_photo_with_metadata(self, file_path=None, fast_mode=False):
        """Capture a photo and return metadata like the dashboard API."""
        if file_path is None:
            # Use FileManager to get consistent naming
            file_manager = FileManager(SAVE_PATH, PROCESSED_PATH)
            file_path = file_manager.get_new_file_path(SAVE_PATH, ORIGINAL_CAPTURE_EXTENSION)

        try:
            self.capture_photo(file_path, fast_mode=fast_mode)

            # Get file info
            file_size = os.path.getsize(file_path)

            return {
                "success": True,
                "photo_id": os.path.splitext(os.path.basename(file_path))[0],
                "original_path": file_path,
                "processed_path": None,  # Will be set after processing
                "file_size": file_size,
                "message": "Photo captured successfully"
            }
        except Exception as e:
            logging.error(f"Error capturing photo: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Photo capture failed: {str(e)}"
            }

    def capture_image_with_metadata(self, file_path=None, fast_mode=False):
        """Capture a PIL image in memory and return metadata for async saving."""
        if file_path is None:
            file_manager = FileManager(SAVE_PATH, PROCESSED_PATH)
            file_path = file_manager.get_new_file_path(SAVE_PATH, ORIGINAL_CAPTURE_EXTENSION)

        try:
            image = self.capture_image(fast_mode=fast_mode)
            return {
                "success": True,
                "photo_id": os.path.splitext(os.path.basename(file_path))[0],
                "original_path": file_path,
                "processed_path": None,
                "file_size": None,
                "message": "Photo captured successfully"
            }, image
        except Exception as e:
            logging.error(f"Error capturing photo: {e}")
            update_diagnostics("camera", "Camera capture failed", "Photo capture failed")
            return {
                "success": False,
                "error": str(e),
                "message": f"Photo capture failed: {str(e)}"
            }, None


    def configure_camera(self):
        """Configure the camera settings."""
        camera_settings = self.settings.get("camera", {})
        resolution = camera_settings.get("resolution", {"width": 1200, "height": 800})

        # Safely stop the camera before reconfiguring to avoid runtime errors
        try:
            self.picam2.stop()
        except Exception:
            pass

        camera_config = self.picam2.create_still_configuration(
            main={"size": (resolution["width"], resolution["height"])}
        )

        # Build controls dictionary from settings
        controls = {
            "ExposureValue": camera_settings.get("exposure_value", 0),
            "Sharpness": camera_settings.get("sharpness", 3)
        }


        camera_config["controls"] = controls

        try:
            self.picam2.configure(camera_config)
        except Exception as e:
            logging.error(f"Error configuring camera: {e}")
            # Try with basic configuration without custom controls
            basic_config = self.picam2.create_still_configuration(
                main={"size": (resolution["width"], resolution["height"])}
            )
            self.picam2.configure(basic_config)
            logging.info("Using basic camera configuration")

        # Set autofocus mode safely
        try:
            af_mode = camera_settings.get("autofocus_mode", 2)
            self.picam2.set_controls({"AfMode": af_mode})
        except Exception as e:
            logging.warning(f"Could not set autofocus mode: {e}")

        self.picam2.start()

    def update_activity_time(self):
        """Update the last activity timestamp."""
        self.last_activity_monotonic = time.monotonic()

    def get_inactivity_seconds(self):
        """Return elapsed inactivity without being affected by clock corrections."""
        return max(0, time.monotonic() - self.last_activity_monotonic)

    def is_timeout_enabled(self):
        """Check if auto-timeout is enabled in settings."""
        return self.settings.get("system", {}).get("auto_timeout_enabled", True)

    def get_timeout_minutes(self):
        """Get the timeout duration in minutes from settings."""
        return self.settings.get("system", {}).get("auto_timeout_minutes", 10)

    def is_timeout_exceeded(self):
        """Check if the timeout period has been exceeded."""
        if not self.is_timeout_enabled():
            return False

        timeout_seconds = self.get_timeout_minutes() * 60
        elapsed = self.get_inactivity_seconds()
        return elapsed > timeout_seconds

    def shutdown_system(self):
        """Safely shutdown the entire Raspberry Pi system to save battery."""
        try:
            timeout_minutes = self.get_timeout_minutes()
            logging.info(f"System has been inactive for {timeout_minutes} minutes. Shutting down to save battery...")
            logging.info("To use the camera again, manually power on the Raspberry Pi")
            # Give a moment for logging to flush
            time.sleep(2)
            # Execute system shutdown command
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
            return True
        except Exception as e:
            logging.error(f"Error shutting down system: {e}")
            return False

    def capture_photo(self, file_path, fast_mode=False):
        """Capture a photo and save it to the specified file path."""
        self._settle_autofocus(fast_mode)
        self.picam2.capture_file(file_path)  # Capture the photo
        logging.info(f"Photo saved to {file_path}")

    def capture_image(self, fast_mode=False):
        """Capture a photo into memory as a PIL image."""
        self._settle_autofocus(fast_mode)
        image = self.picam2.capture_image("main")
        logging.info("Photo captured to memory")
        return image

    def _settle_autofocus(self, fast_mode=False):
        """Give autofocus a short settle window before capture."""
        self.update_activity_time()
        autofocus_mode = self.settings.get("camera", {}).get("autofocus_mode", 2)
        if fast_mode:
            sleep(0.1)  # Shorter autofocus for startup
            logging.info("Fast autofocus mode: 0.1s delay")
        elif self._has_captured and autofocus_mode == 2:
            logging.info("Continuous autofocus already active: no settle delay")
        elif self._has_captured:
            sleep(0.1)  # Sensor already focused from previous capture
            logging.info("Adaptive autofocus: 0.1s delay (sensor pre-focused)")
        else:
            sleep(0.3)  # First capture needs full autofocus settle time
        self._has_captured = True

class ImageProcessor:
    """Handles image processing, including resizing, dithering, and saving."""

    @staticmethod
    def get_bayer_matrix(size):
        """Generate Bayer matrix for ordered dithering."""
        np = _lazy_import_numpy()
        if size == 2:
            return np.array([[0, 2], [3, 1]], dtype=np.float32) / 4.0
        elif size == 4:
            return np.array([
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5]
            ], dtype=np.float32) / 16.0
        elif size == 8:
            return np.array([
                [0, 32, 8, 40, 2, 34, 10, 42],
                [48, 16, 56, 24, 50, 18, 58, 26],
                [12, 44, 4, 36, 14, 46, 6, 38],
                [60, 28, 52, 20, 62, 30, 54, 22],
                [3, 35, 11, 43, 1, 33, 9, 41],
                [51, 19, 59, 27, 49, 17, 57, 25],
                [15, 47, 7, 39, 13, 45, 5, 37],
                [63, 31, 55, 23, 61, 29, 53, 21]
            ], dtype=np.float32) / 64.0
        else:
            # Default to 4x4 if unsupported size
            return ImageProcessor.get_bayer_matrix(4)

    @staticmethod
    def apply_ordered_dithering(image, saturation=0.6, brightness_factor=1.1, color_factor=1.4,
                               bayer_size=4, threshold_scale=1.0):
        """Apply ordered dithering using standard Bayer threshold + LUT nearest-color."""
        Image, ImageEnhance = _lazy_import_pil()
        np = _lazy_import_numpy()

        import time
        start_time = time.monotonic()

        # Ensure the image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Adjust brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)

        # Adjust color intensity
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(color_factor)

        # Convert image to numpy array
        img_array = np.array(image, dtype=np.float32)
        height, width = img_array.shape[:2]

        # Get Bayer matrix and tile it to cover the entire image
        bayer_matrix = ImageProcessor.get_bayer_matrix(bayer_size)
        y_tiles = (height + bayer_size - 1) // bayer_size
        x_tiles = (width + bayer_size - 1) // bayer_size
        threshold_matrix = np.tile(bayer_matrix, (y_tiles, x_tiles))[:height, :width]

        # Build the blended palette (same as Floyd-Steinberg)
        palette_colors = []
        color_indices = [0, 1, 5, 4, 0, 3, 2]
        for i in color_indices:
            rs, gs, bs = [c * saturation for c in SATURATED_PALETTE[i]]
            rd, gd, bd = [c * (1.0 - saturation) for c in DESATURATED_PALETTE[i]]
            palette_colors.append([int(rs + rd), int(gs + gd), int(bs + bd)])
        pal = np.array(palette_colors, dtype=np.float32)

        # --- Precompute 32K nearest-color LUT (CIELAB distance) ---
        # CIELAB properly separates lightness from chromaticity, preventing
        # neutral grays from matching green (which has similar luminance).
        def _rgb_to_lab_batch(rgb):
            """Convert (N,3) float32 RGB [0-255] to CIELAB. Fully vectorized."""
            c = rgb / 255.0
            linear = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
            M = np.array([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
            xyz = linear @ M.T
            xyz[:, 0] /= 0.95047
            xyz[:, 2] /= 1.08883
            f = np.where(xyz > 0.008856, np.cbrt(xyz), (903.3 * xyz + 16.0) / 116.0)
            lab = np.empty_like(xyz)
            lab[:, 0] = 116.0 * f[:, 1] - 16.0
            lab[:, 1] = 500.0 * (f[:, 0] - f[:, 1])
            lab[:, 2] = 200.0 * (f[:, 1] - f[:, 2])
            return lab

        # Build all possible 5-bit RGB values (32 levels per channel)
        r_vals = np.arange(32, dtype=np.float32) * 8 + 4
        g_vals = np.arange(32, dtype=np.float32) * 8 + 4
        b_vals = np.arange(32, dtype=np.float32) * 8 + 4
        rr, gg, bb = np.meshgrid(r_vals, g_vals, b_vals, indexing='ij')
        all_rgb = np.stack([rr, gg, bb], axis=-1).reshape(-1, 3)  # (32768, 3)

        # Convert to CIELAB
        all_lab = _rgb_to_lab_batch(all_rgb)
        pal_lab = _rgb_to_lab_batch(pal)

        # Compute CIELAB ΔE² to each palette color
        # all_lab: (32768, 3), pal_lab: (7, 3) → distances: (32768, 7)
        diff = all_lab[:, np.newaxis, :] - pal_lab[np.newaxis, :, :]
        distances = np.sum(diff * diff, axis=2)
        nearest_lut = np.argmin(distances, axis=1).astype(np.uint8)  # (32768,)

        lut_time = time.monotonic()
        logging.info(f"LUT built in {(lut_time - start_time)*1000:.1f}ms")

        # --- Per-channel Bayer thresholds derived from palette spacing ---
        # Compute max gap between successive sorted values per channel
        thresholds = np.zeros(3, dtype=np.float32)
        for ch in range(3):
            vals = np.sort(pal[:, ch])
            if len(vals) > 1:
                gaps = np.diff(vals)
                thresholds[ch] = float(np.max(gaps))
            else:
                thresholds[ch] = 256.0
        thresholds *= threshold_scale

        # --- Apply standard ordered dithering (Bayer noise + nearest color) ---
        # For each pixel: Attempt = Input + (threshold_value - 0.5) * Threshold
        # Then find nearest color from palette
        noise_r = (threshold_matrix - 0.5) * thresholds[0]
        noise_g = (threshold_matrix - 0.5) * thresholds[1]
        noise_b = (threshold_matrix - 0.5) * thresholds[2]

        dithered = np.empty_like(img_array)
        dithered[:, :, 0] = np.clip(img_array[:, :, 0] + noise_r, 0, 255)
        dithered[:, :, 1] = np.clip(img_array[:, :, 1] + noise_g, 0, 255)
        dithered[:, :, 2] = np.clip(img_array[:, :, 2] + noise_b, 0, 255)

        # Quantize to 5 bits and look up nearest color from LUT
        dithered_q = (dithered / 8).astype(np.int32)
        dithered_q = np.clip(dithered_q, 0, 31)
        lut_keys = (dithered_q[:, :, 0] << 10) | (dithered_q[:, :, 1] << 5) | dithered_q[:, :, 2]
        output_array = nearest_lut[lut_keys].astype(np.uint8)

        # Build output palette image
        palette_flat = []
        for color in palette_colors:
            palette_flat.extend(color)
        palette_flat += [0, 0, 0] * (256 - len(palette_colors))

        output_image = Image.fromarray(output_array, mode='P')
        output_image.putpalette(palette_flat)

        end_time = time.monotonic()
        logging.info(f"Ordered dithering completed in {(end_time - start_time)*1000:.1f}ms for {height}x{width} image")

        return output_image

    @staticmethod
    def palette_blend(saturation, dtype='uint8'):
        """Blend between desaturated and saturated palettes based on saturation."""
        palette = []
        color_indices = [0, 1, 5, 4, 0, 3, 2]
        for i in color_indices:
            rs, gs, bs = [c * saturation for c in SATURATED_PALETTE[i]]
            rd, gd, bd = [c * (1.0 - saturation) for c in DESATURATED_PALETTE[i]]
            if dtype == 'uint8':
                palette += [int(rs + rd), int(gs + gd), int(bs + bd)]
            elif dtype == 'uint24':
                palette += [(int(rs + rd) << 16) | (int(gs + gd) << 8) | int(bs + bd)]
        return palette

    @staticmethod
    def apply_dithering(image, saturation=0.6, brightness_factor=1.1, color_factor=1.4,
                       dithering_method="floyd_steinberg", bayer_size=4, threshold_scale=1.0):
        """Applies brightness, color enhancement, and dithering."""
        if dithering_method == "ordered":
            return ImageProcessor.apply_ordered_dithering(
                image, saturation, brightness_factor, color_factor, bayer_size, threshold_scale
            )
        else:
            Image, ImageEnhance = _lazy_import_pil()

            # Default Floyd-Steinberg dithering
            # Ensure the image is in RGB mode
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Adjust brightness
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_factor)

            # Adjust saturation
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(color_factor)

            # Blend the palette
            palette = ImageProcessor.palette_blend(saturation)

            # Create a new palette image
            palette_image = Image.new("P", (1, 1))
            palette_image.putpalette(palette + [0, 0, 0] * (256 - len(palette) // 3))

            # Convert the image using the custom palette and Floyd-Steinberg dithering
            converted_image = image.quantize(palette=palette_image, dither=Image.FLOYDSTEINBERG)

            return converted_image

    @staticmethod
    def dither_to_display_buffer(image, saturation=0.6, brightness_factor=1.1, color_factor=1.4,
                                  dithering_method="floyd_steinberg", bayer_size=4, threshold_scale=1.0):
        """Dither an image and produce the packed display buffer in one step.

        Returns a tuple of (display_buffer, dithered_pil_image):
        - display_buffer: bytearray ready to send to epd.display()
        - dithered_pil_image: the palette PIL image (for saving as PNG for the dashboard)

        This eliminates the redundant img2buffer() conversion by mapping dither
        palette indices directly to hardware nibble indices during the dither step.
        """
        np = _lazy_import_numpy()
        Image, _ = _lazy_import_pil()
        import time
        start_time = time.monotonic()

        # Mapping from dither palette indices to hardware nibble indices.
        # Dither palette: 0:black, 1:white, 2:yellow, 3:red, 4:black, 5:blue, 6:green
        # Hardware panel:  0:black, 1:white, 2:yellow, 3:red, (skip 4), 5:blue, 6:green
        # Indices 7-255 are black padding in the palette, so they map to hw black (0).
        DITHER_TO_HW = np.zeros(256, dtype=np.uint8)  # default=0 (black) for all padding
        DITHER_TO_HW[0] = 0   # black  → hw black
        DITHER_TO_HW[1] = 1   # white  → hw white
        DITHER_TO_HW[2] = 2   # yellow → hw yellow
        DITHER_TO_HW[3] = 3   # red    → hw red
        DITHER_TO_HW[4] = 0   # black  → hw black (duplicate)
        DITHER_TO_HW[5] = 5   # blue   → hw blue
        DITHER_TO_HW[6] = 6   # green  → hw green

        # Step 1: run the normal dithering to get the palette image
        dithered_image = ImageProcessor.apply_dithering(
            image, saturation, brightness_factor, color_factor,
            dithering_method, bayer_size, threshold_scale
        )

        dither_time = time.monotonic()

        # Step 2: extract pixel indices and rotate if needed.
        # The processed image is landscape; the current panel expects portrait.
        imwidth, imheight = dithered_image.size
        if (imwidth, imheight) == DISPLAY_IMAGE_SIZE:
            display_image = dithered_image.rotate(90, expand=True)
        elif (imwidth, imheight) == DISPLAY_PANEL_SIZE:
            display_image = dithered_image
        else:
            logging.warning(f"dither_to_display_buffer: unexpected size {imwidth}x{imheight}")
            display_image = dithered_image.rotate(90, expand=True)

        # Step 3: map palette indices → hardware nibbles and pack
        src_indices = np.frombuffer(display_image.tobytes('raw'), dtype=np.uint8)
        hw_pixels = DITHER_TO_HW[src_indices]
        buf = (hw_pixels[0::2].astype(np.uint8) << 4) + hw_pixels[1::2].astype(np.uint8)
        display_buffer = bytearray(buf.astype(np.uint8))

        buffer_time = time.monotonic()
        logging.info(f"dither_to_display_buffer: dither={((dither_time - start_time)*1000):.0f}ms, "
                     f"buffer={((buffer_time - dither_time)*1000):.0f}ms, "
                     f"total={((buffer_time - start_time)*1000):.0f}ms")

        return display_buffer, dithered_image

    @staticmethod
    def resize_image(image, size=DISPLAY_IMAGE_SIZE):
        """Resizes the image to the specified size."""
        Image, _ = _lazy_import_pil()
        if image.size == (size[0] * 2, size[1] * 2) and hasattr(image, "reduce"):
            # Preserve the 2x source capture and average each 2x2 block before
            # dithering. This is faster than generic scaling and still provides
            # a properly filtered high-resolution source.
            return image.reduce(2)

        # Pillow >= 10 moved filters under Image.Resampling; older Pi builds keep them on Image.
        resampling_attr = getattr(Image, "Resampling", Image)
        resample_filter = getattr(resampling_attr, "BILINEAR", Image.BILINEAR)
        return image.resize(size, resample_filter)

    @staticmethod
    def img2buffer(image, width=DISPLAY_PANEL_WIDTH, height=DISPLAY_PANEL_HEIGHT):
        """Converts an image to a format suitable for the e-ink display."""
        np = _lazy_import_numpy()
        imwidth, imheight = image.size
        if imwidth == width and imheight == height:
            image_temp = image
        elif imwidth == height and imheight == width:
            image_temp = image.rotate(90, expand=True)
        else:
            logging.warning(f"Invalid image dimensions: {imwidth}x{imheight}, expected {width}x{height}")
            return None

        # Ensure PIL is available for palette operations in this function
        Image, _ = _lazy_import_pil()

        # Map any palette indices or RGB values to the panel's fixed nibble indices.
        # Hardware palette indices expected by the panel (nibbles):
        # 0:black, 1:white, 2:yellow, 3:red, 4:clear/duplicate-black (do not use), 5:blue, 6:green
        # We'll map every pixel to the closest of {0,1,2,3,5,6} and never emit 4.
        try:
            # Ensure we have a palette image to read indices from
            if image_temp.mode != 'P':
                pal_image = Image.new('P', (1, 1))
                # Build palette matching driver order so indices match hardware mapping
                pal_image.putpalette([
                    0, 0, 0,      # 0 black
                    255, 255, 255,# 1 white
                    255, 255, 0,  # 2 yellow
                    255, 0, 0,    # 3 red
                    0, 0, 0,      # 4 duplicate black (panel clear)
                    0, 0, 255,    # 5 blue
                    0, 255, 0,    # 6 green
                ] + [0, 0, 0] * (256 - 7))
                # Quantize without dithering to preserve the existing pattern as much as possible
                image_temp = image_temp.convert('RGB').quantize(palette=pal_image, dither=Image.NONE)

            # Build source palette -> hardware index map using nearest color, excluding index 4
            pal = image_temp.getpalette()
            if pal is None:
                raise ValueError('Palette missing after conversion to P')
            src_colors = [pal[i:i+3] for i in range(0, min(len(pal), 256 * 3), 3)]

            # Hardware color set (exclude index 4)
            hw_indices = np.array([0, 1, 2, 3, 5, 6], dtype=np.uint8)
            hw_colors = np.array([
                [0, 0, 0],
                [255, 255, 255],
                [255, 255, 0],
                [255, 0, 0],
                [0, 0, 255],
                [0, 255, 0],
            ], dtype=np.float32)

            # Create a 256-element map from source palette index to hardware nibble index
            idx_map = np.zeros(256, dtype=np.uint8)
            for s_idx in range(256):
                if s_idx < len(src_colors):
                    r, g, b = src_colors[s_idx]
                    color_vec = np.array([[float(r), float(g), float(b)]], dtype=np.float32)
                    dists = np.sum((hw_colors - color_vec) ** 2, axis=1)
                    mapped = int(hw_indices[int(np.argmin(dists))])
                    # Never allow 4; nearest set excludes 4 already. Keep mapped as is.
                    idx_map[s_idx] = np.uint8(mapped)
                else:
                    # Uninitialized palette slots default to white
                    idx_map[s_idx] = 1

            # Apply mapping to pixel indices and pack nibbles
            src_indices = np.frombuffer(image_temp.tobytes('raw'), dtype=np.uint8)
            hw_pixels = idx_map[src_indices]
            # Double safety: remap any stray 4 -> 0
            if (hw_pixels == 4).any():
                hw_pixels = np.where(hw_pixels == 4, 0, hw_pixels).astype(np.uint8)
            buf = (hw_pixels[0::2].astype(np.uint8) << 4) + hw_pixels[1::2].astype(np.uint8)
            buf = buf.astype(np.uint8).tolist()
            return buf
        except Exception as e:
            logging.warning(f"img2buffer: palette mapping fallback due to: {e}")
            # Fallback: direct indices with 4 -> 0 remap
            buf_6color = np.frombuffer(image_temp.tobytes('raw'), dtype=np.uint8)
            try:
                buf_6color = buf_6color.copy()
                buf_6color[buf_6color == 4] = 0
            except Exception:
                buf_6color = np.where(buf_6color == 4, 0, buf_6color).astype(np.uint8)
            buf = (buf_6color[0::2] << 4) + buf_6color[1::2]
            buf = buf.astype(np.uint8).tolist()

        return buf

    @staticmethod
    def process_photo_with_settings(original_path, output_path, processing_settings):
        """Process a photo with specific settings and save it."""
        try:
            Image, ImageEnhance = _lazy_import_pil()

            # Load original image
            original_image = Image.open(original_path)

            # Resize image
            resized_image = ImageProcessor.resize_image(original_image)

            # Apply processing with settings
            dithered_image = ImageProcessor.apply_dithering(
                resized_image,
                saturation=processing_settings.get("saturation", 0.6),
                brightness_factor=processing_settings.get("brightness_factor", 1.1),
                color_factor=processing_settings.get("color_factor", 1.4),
                dithering_method=processing_settings.get("dithering_method", "floyd_steinberg"),
                bayer_size=processing_settings.get("bayer_size", 4),
                threshold_scale=processing_settings.get("threshold_scale", 1.0)
            )

            # Save processed image as PNG (keep palette if present)
            dithered_image.save(output_path, format="PNG")
            logging.info(f"Processed image saved to {output_path}")

            return {
                "success": True,
                "original_path": original_path,
                "processed_path": output_path,
                "message": "Photo processed successfully"
            }

        except Exception as e:
            logging.error(f"Error processing photo: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Photo processing failed: {str(e)}"
            }

    @staticmethod
    def reprocess_photo_by_id(photo_id, processing_settings, photos_path=SAVE_PATH, output_path=PROCESSED_PATH):
        """Reprocess an existing photo by ID with new settings."""
        # Find the original photo
        original_path = None
        for ext in ['png', 'jpg', 'jpeg']:
            test_path = os.path.join(photos_path, f"{photo_id}.{ext}")
            if os.path.exists(test_path):
                original_path = test_path
                break

        if not original_path:
            return {
                "success": False,
                "error": "Original photo not found",
                "message": f"Could not find original photo for ID: {photo_id}"
            }

        # Generate output path
        output_file_path = os.path.join(output_path, f"{photo_id}_dithered.png")

        return ImageProcessor.process_photo_with_settings(original_path, output_file_path, processing_settings)


class FileManager:
    """Handles file saving and directory management."""

    def __init__(self, save_path, processed_path):
        self.save_path = save_path
        self.processed_path = processed_path
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(processed_path, exist_ok=True)
        self._id_lock = threading.Lock()
        self._next_photo_index = self._find_next_photo_index()

    def _find_next_photo_index(self):
        """Seed the monotonic photo counter from numeric filenames on disk."""
        highest_index = -1
        try:
            for filename in os.listdir(self.save_path):
                stem, extension = os.path.splitext(filename)
                if extension.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                if stem.isdigit():
                    highest_index = max(highest_index, int(stem))
        except OSError as e:
            logging.warning(f"Could not scan existing photo IDs: {e}")
        return highest_index + 1

    def get_new_file_path(self, folder, extension="png"):
        """Generates a new unique file path in the specified folder."""
        if os.path.abspath(folder) != os.path.abspath(self.save_path):
            raise ValueError("Photo IDs can only be allocated in the original photo directory")

        with self._id_lock:
            index = self._next_photo_index
            self._next_photo_index += 1
        return os.path.join(folder, f"{str(index).zfill(5)}.{extension}")

    def save_image(self, image, folder, extension="png"):
        """Saves the image to a unique file in the specified folder."""
        file_path = self.get_new_file_path(folder, extension)
        # Always save PNG by default
        image.save(file_path, format="PNG")
        logging.info(f"Image saved to {file_path}")
        return file_path

    def get_photo_info(self, photo_id):
        """Get information about a specific photo by ID."""
        # Find original photo
        original_path = None
        for ext in ['png', 'jpg', 'jpeg']:
            test_path = os.path.join(self.save_path, f"{photo_id}.{ext}")
            if os.path.exists(test_path):
                original_path = test_path
                break

        if not original_path:
            return None

        try:
            # Check for dithered version
            dithered_path = os.path.join(self.processed_path, f"{photo_id}_dithered.png")
            has_dithered = os.path.exists(dithered_path)

            # Get file stats
            original_stat = os.stat(original_path)

            return {
                "id": photo_id,
                "original_path": original_path,
                "dithered_path": dithered_path if has_dithered else None,
                "has_dithered": has_dithered,
                "file_size": original_stat.st_size,
                "created_at": original_stat.st_mtime,
                "filename": os.path.basename(original_path)
            }
        except OSError as e:
            logging.warning(f"Could not stat photo {photo_id} (may be mid-write): {e}")
            return None

    def list_all_photos(self):
        """List all photos with their information."""
        photos = []

        # Get all files from the save directory
        try:
            files = os.listdir(self.save_path)
            photo_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for filename in sorted(photo_files, reverse=True):  # Newest first
                photo_id = os.path.splitext(filename)[0]
                photo_info = self.get_photo_info(photo_id)
                if photo_info:
                    photos.append(photo_info)

        except Exception as e:
            logging.error(f"Error listing photos: {e}")

        return photos

    def delete_photo(self, photo_id):
        """Delete both original and processed versions of a photo."""
        deleted_files = []

        # Delete original
        for ext in ['png', 'jpg', 'jpeg']:
            original_path = os.path.join(self.save_path, f"{photo_id}.{ext}")
            if os.path.exists(original_path):
                os.remove(original_path)
                deleted_files.append(original_path)
                break

        # Delete processed version
        # Delete processed version (both png and legacy jpg)
        dithered_png = os.path.join(self.processed_path, f"{photo_id}_dithered.png")
        dithered_jpg = os.path.join(self.processed_path, f"{photo_id}_dithered.jpg")
        for p in (dithered_png, dithered_jpg):
            if os.path.exists(p):
                os.remove(p)
                deleted_files.append(p)

        if deleted_files:
            logging.info(f"Deleted photo {photo_id}: {deleted_files}")
            return {"success": True, "deleted_files": deleted_files}
        else:
            return {"success": False, "error": "Photo not found"}


# ═══════════════════════════════════════════════════════════════════
# HARDWARE: Display driver wrapper
# This class adapts the Waveshare epd4in0e driver to the rest of reFrame.
# To use another display, keep this public surface compatible:
# prepare_async(), is_busy(), display_image(), display_buffer(),
# display_buffer_async(), display_photo_by_id(), clear_display(),
# display_dashboard_qr(), and sleep().
#
# Also update DISPLAY_* constants and the dither palette/mapping in
# ImageProcessor if the panel has a different resolution, orientation, color
# order, or buffer format.
# ═══════════════════════════════════════════════════════════════════
class EInkDisplay:
    """Waveshare ePaper display adapter with lazy initialization."""

    def __init__(self, settings_path="settings.json"):
        # Don't initialize e-ink hardware at startup — takes 5-10s
        self.settings_path = settings_path
        self.epd = None
        self._initialized = False
        self._display_busy = False  # True while the panel is mid-refresh
        self._display_thread = None
        self._display_lock = threading.Lock()
        self._init_lock = threading.Lock()
        self._init_thread = None
        logging.info("E-ink display: Lazy initialization enabled")

    def _ensure_initialized(self):
        """Initialize e-ink display only when first needed."""
        with self._init_lock:
            if self._initialized:
                return
            try:
                logging.info("Initializing e-ink display hardware...")
                import time
                from waveshare_epd import epd4in0e
                start_time = time.monotonic()

                self.epd = epd4in0e.EPD()
                self.epd.init()

                init_time = time.monotonic() - start_time
                logging.info(f"E-ink display ready in {init_time:.2f}s")
                self._initialized = True
            except Exception as error:
                logging.error("Display initialization failed: %s", error)
                update_diagnostics(
                    "display",
                    "Display failed to initialize. Check the display connection and SPI.",
                    "Display initialization failed",
                )
                raise

    def prepare_async(self):
        """Start e-ink hardware initialization in the background."""
        if self._initialized:
            return
        if self._init_thread and self._init_thread.is_alive():
            return
        self._init_thread = threading.Thread(target=self._ensure_initialized, daemon=True)
        self._init_thread.start()

    def is_busy(self):
        """Check if the display is currently in the middle of a refresh cycle."""
        return self._display_busy

    def _fast_refresh_enabled(self):
        """Snapshot the saved preference before a refresh, never mid-waveform."""
        try:
            with open(self.settings_path, 'r') as settings_file:
                settings = json.load(settings_file)
            display_settings = settings.get("display", {}) if isinstance(settings, dict) else {}
            return isinstance(display_settings, dict) and display_settings.get("fast_refresh") is True
        except (OSError, ValueError):
            return False

    def display_image(self, image):
        """Displays the provided image on the e-ink display."""
        buffer = ImageProcessor.img2buffer(image)
        self.display_buffer(buffer)

    def display_buffer(self, buffer):
        """Send a pre-built display buffer to the e-ink panel (blocking)."""
        self._ensure_initialized()
        if buffer:
            self.epd.display(buffer, fast_refresh=self._fast_refresh_enabled())

    def display_buffer_async(self, buffer):
        """Send a pre-built display buffer to the e-ink panel in a background thread.

        Sets _display_busy=True before starting and clears it when the refresh
        cycle completes. The caller should check is_busy() before starting a
        new capture to avoid invisible captures with no feedback.
        """
        if not buffer:
            return

        with self._display_lock:
            # If a previous refresh is somehow still running, log a warning
            if self._display_busy:
                logging.warning("display_buffer_async: previous refresh still in progress, skipping")
                return
            self._display_busy = True

        def _refresh():
            try:
                self._ensure_initialized()
                logging.info("Display refresh started (background)")
                refresh_start = time.monotonic()
                self.epd.display(buffer, fast_refresh=self._fast_refresh_enabled())
                refresh_time = time.monotonic() - refresh_start
                logging.info(f"Display refresh completed in {refresh_time:.1f}s")
                update_diagnostics("display", activity="Display refresh completed")
            except Exception as e:
                logging.error(f"Display refresh error: {e}")
                update_diagnostics(
                    "display",
                    "Display refresh failed. Check the display connection and SPI.",
                    "Display refresh failed",
                )
            finally:
                with self._display_lock:
                    self._display_busy = False

        self._display_thread = threading.Thread(target=_refresh, daemon=True)
        self._display_thread.start()

    def display_photo_by_id(self, photo_id, file_manager, prefer_dithered=True):
        """Display a photo by ID on the e-ink screen."""
        try:
            photo_info = file_manager.get_photo_info(photo_id)
            if not photo_info:
                return {
                    "success": False,
                    "error": "Photo not found",
                    "message": f"Could not find photo with ID: {photo_id}"
                }

            # Choose which version to display
            if prefer_dithered and photo_info["has_dithered"]:
                image_path = photo_info["dithered_path"]
                version = "dithered"
            else:
                image_path = photo_info["original_path"]
                version = "original"

            # Load and display the image
            Image, ImageEnhance = _lazy_import_pil()
            image = Image.open(image_path)

            # If displaying original, we need to process it first
            if version == "original":
                resized_image = ImageProcessor.resize_image(image)
                # Use default processing settings for display
                display_image = ImageProcessor.apply_dithering(resized_image)
                self.display_image(display_image)
            else:
                # Dithered image can be displayed directly (just resize if needed)
                if image.size != DISPLAY_IMAGE_SIZE:
                    image = ImageProcessor.resize_image(image)
                self.display_image(image)

            logging.info(f"Displayed {version} version of photo {photo_id} on e-ink screen")

            return {
                "success": True,
                "photo_id": photo_id,
                "version_displayed": version,
                "image_path": image_path,
                "message": f"Photo {photo_id} displayed successfully"
            }

        except Exception as e:
            logging.error(f"Error displaying photo {photo_id}: {e}")
            update_diagnostics("display", "Display refresh failed", "Display refresh failed")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to display photo {photo_id}: {str(e)}"
            }

    def clear_display(self):
        """Clear the e-ink display."""
        self._ensure_initialized()  # Initialize only when first used
        try:
            self.epd.Clear(fast_refresh=self._fast_refresh_enabled())
            logging.info("E-ink display cleared")
            update_diagnostics("display", activity="Display cleared")
            return {"success": True, "message": "Display cleared"}
        except Exception as e:
            logging.error(f"Error clearing display: {e}")
            update_diagnostics("display", "Display clear failed", "Display clear failed")
            return {"success": False, "error": str(e)}

    def display_dashboard_qr(self, access_info):
        """Display a QR code that opens the dashboard."""
        try:
            image = render_dashboard_qr_image(access_info)
            self.display_image(image)
            logging.info("Displayed dashboard QR code: %s", access_info.get("primary_url"))
            update_diagnostics("display", activity="Dashboard QR displayed")
            return {
                "success": True,
                "message": "Dashboard QR displayed",
                "access": access_info
            }
        except Exception as e:
            logging.error(f"Error displaying dashboard QR: {e}")
            update_diagnostics("display", "Display refresh failed", "Display refresh failed")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to display dashboard QR: {str(e)}"
            }

    def sleep(self):
        """Puts the e-ink display to sleep."""
        if self._initialized and self.epd:
            self.epd.sleep()


class CameraSystem:
    """Complete camera system that implements dashboard-like functionality."""

    def __init__(self, settings_path="settings.json", eink_display=None):
        self.eink_display = eink_display if eink_display is not None else EInkDisplay(settings_path)
        self.camera_manager = CameraManager(settings_path)
        self.file_manager = FileManager(SAVE_PATH, PROCESSED_PATH)
        self.timeout_thread = None
        self.timeout_running = False
        self._timeout_started = False
        self._timeout_stop_event = threading.Event()
        self.dashboard_qr_thread = None
        self._dashboard_qr_monitor_started = False
        self.dashboard_qr_running = False
        logging.info("Timeout monitor initialization deferred for fast startup")

    def start_timeout_monitor(self):
        """Start the background timeout monitoring thread."""
        if not self._timeout_started and self.camera_manager.is_timeout_enabled():
            self.timeout_running = True
            self._timeout_stop_event.clear()
            self.timeout_thread = threading.Thread(target=self._timeout_monitor_loop, daemon=True)
            self.timeout_thread.start()
            self._timeout_started = True
            timeout_minutes = self.camera_manager.get_timeout_minutes()
            elapsed = self.camera_manager.get_inactivity_seconds()
            logging.info(f"Auto-timeout monitor started: {timeout_minutes} minutes timeout, last activity was {elapsed:.1f}s ago")

    def start_timeout_monitor_deferred(self):
        """Start the timeout monitor after first photo for faster startup."""
        if not self._timeout_started:
            logging.info("Starting deferred timeout monitor...")
            self.start_timeout_monitor()

    def stop_timeout_monitor(self):
        """Stop the background timeout monitoring thread."""
        self.timeout_running = False
        self._timeout_stop_event.set()
        if self.timeout_thread and self.timeout_thread.is_alive():
            self.timeout_thread.join(timeout=2)
        self._timeout_started = False

    def _timeout_monitor_loop(self):
        """Background loop that checks for timeout and shuts down system."""
        # Wait a bit before starting to check for timeout to avoid false triggers on startup
        if self._timeout_stop_event.wait(60):
            return

        while self.timeout_running and not self._timeout_stop_event.is_set():
            try:
                if self.camera_manager.is_timeout_exceeded():
                    logging.info("Timeout exceeded, initiating system shutdown")
                    self.camera_manager.shutdown_system()
                    # If we reach here, shutdown failed, so stop monitoring
                    break

                # Check every 30 seconds
                for _ in range(30):
                    if self._timeout_stop_event.wait(1):
                        break

            except Exception as e:
                logging.error(f"Error in timeout monitor: {e}")
                if self._timeout_stop_event.wait(10):
                    break

    def update_activity(self):
        """Update activity time."""
        self.camera_manager.update_activity_time()

    def get_dashboard_access_api(self):
        """API-style dashboard access information."""
        return {
            "success": True,
            "access": get_dashboard_access_info()
        }

    def display_dashboard_qr_api(self):
        """API-style dashboard QR display."""
        access_info = get_dashboard_access_info()
        if not access_info.get("ip_address"):
            return {
                "success": False,
                "message": "No usable LAN IP address found yet",
                "access": access_info
            }

        result = self.eink_display.display_dashboard_qr(access_info)
        return result

    def start_dashboard_qr_monitor(self):
        """Watch for usable network changes for the lifetime of the process."""
        if self._dashboard_qr_monitor_started:
            return

        self._dashboard_qr_monitor_started = True
        self.dashboard_qr_running = True
        self.dashboard_qr_thread = threading.Thread(target=self._dashboard_qr_monitor_loop, daemon=True)
        self.dashboard_qr_thread.start()

    def stop_dashboard_qr_monitor(self):
        self.dashboard_qr_running = False
        if self.dashboard_qr_thread and self.dashboard_qr_thread.is_alive():
            self.dashboard_qr_thread.join(timeout=1)

    def _dashboard_qr_monitor_loop(self):
        last_connection_ip = None
        was_enabled = False
        while self.dashboard_qr_running:
            try:
                system_settings = self.camera_manager.settings.get("system", {})
                enabled = system_settings.get(
                    "show_dashboard_qr_on_wifi_connect",
                    system_settings.get("show_dashboard_qr_on_first_network", True)
                )
                if not enabled:
                    last_connection_ip = get_lan_ip_address()
                    was_enabled = False
                    time.sleep(5)
                    continue

                access_info = get_dashboard_access_info()
                ip_address = access_info.get("ip_address")
                if not ip_address:
                    last_connection_ip = None
                    was_enabled = True
                elif not was_enabled or ip_address != last_connection_ip:
                    if self.eink_display.is_busy():
                        time.sleep(5)
                        continue
                    logging.info(
                        "Wi-Fi connected at %s; displaying dashboard QR",
                        ip_address
                    )
                    with _operation_lock:
                        result = self.display_dashboard_qr_api()
                    if result.get("success"):
                        last_connection_ip = ip_address
                        was_enabled = True
            except Exception as e:
                logging.warning(f"Dashboard QR monitor error: {e}")
            time.sleep(5)

    def capture_photo_api(self, fast_mode=False):
        """API-style photo capture with optimized display pipeline.

        Pipeline: capture to memory → dither+buffer → async display → background file save.
        This minimizes the delay between shutter press and the display starting to refresh.
        """
        try:
            photo_path = self.file_manager.get_new_file_path(SAVE_PATH, ORIGINAL_CAPTURE_EXTENSION)
            logging.info(f"Capturing photo to: {photo_path}")

            display_settings = self.camera_manager.settings.get("display", {})
            if display_settings.get("auto_display", True):
                # Hide first-use GPIO/SPI initialization behind capture and
                # image processing instead of delaying the physical refresh.
                self.eink_display.prepare_async()

            pipeline_start = time.monotonic()
            result, original_image = self.camera_manager.capture_image_with_metadata(photo_path, fast_mode=fast_mode)

            if result["success"]:
                capture_time = time.monotonic()
                logging.info(f"Photo captured successfully: {result['photo_id']} "
                             f"({((capture_time - pipeline_start)*1000):.0f}ms)")
                update_diagnostics("camera", activity="Photo captured successfully")

                processing_settings = self.camera_manager.settings.get("processing", {})
                dithered_path = os.path.join(PROCESSED_PATH, f"{result['photo_id']}_dithered.png")

                resized_image = ImageProcessor.resize_image(original_image)

                # Combined dither + buffer: produces display-ready buffer AND the dithered image
                display_buffer, dithered_image = ImageProcessor.dither_to_display_buffer(
                    resized_image,
                    saturation=processing_settings.get("saturation", 0.6),
                    brightness_factor=processing_settings.get("brightness_factor", 1.1),
                    color_factor=processing_settings.get("color_factor", 1.4),
                    dithering_method=processing_settings.get("dithering_method", "floyd_steinberg"),
                    bayer_size=processing_settings.get("bayer_size", 4),
                    threshold_scale=processing_settings.get("threshold_scale", 1.0)
                )

                process_time = time.monotonic()
                logging.info(f"Dither+buffer complete ({((process_time - capture_time)*1000):.0f}ms)")

                # Send to display ASAP (async — screen starts blinking immediately)
                if display_settings.get("auto_display", True):
                    logging.info("Sending to display (async)")
                    self.eink_display.display_buffer_async(display_buffer)

                display_sent_time = time.monotonic()
                logging.info(f"Total button-to-display: {((display_sent_time - pipeline_start)*1000):.0f}ms")

                # Save files in background after the display refresh has been dispatched.
                result["processed_path"] = dithered_path
                def _save_outputs():
                    try:
                        original_image.save(photo_path, format="JPEG")
                        result["file_size"] = os.path.getsize(photo_path)
                        logging.info(f"Original JPEG saved: {photo_path}")
                        dithered_image.save(dithered_path, format="PNG")
                        logging.info(f"Dithered PNG saved: {dithered_path}")
                        update_diagnostics("storage")
                    except Exception as e:
                        logging.error(f"Error saving photo outputs: {e}")
                        update_diagnostics(
                            "storage",
                            "Captured photos could not be saved",
                            "Photo save failed",
                        )

                save_thread = threading.Thread(target=_save_outputs, daemon=True)
                save_thread.start()

            return result

        except Exception as e:
            logging.error(f"Error in capture_photo_api: {e}")
            update_diagnostics("camera", "Photo capture failed", "Photo capture failed")
            return {
                "success": False,
                "error": str(e),
                "message": f"Photo capture failed: {str(e)}"
            }

    def display_photo_api(self, photo_id):
        """API-style photo display."""
        return self.eink_display.display_photo_by_id(photo_id, self.file_manager)

    def reprocess_photo_api(self, photo_id, processing_settings=None):
        """API-style photo reprocessing."""
        if processing_settings is None:
            processing_settings = self.camera_manager.settings.get("processing", {})

        return ImageProcessor.reprocess_photo_by_id(photo_id, processing_settings)

    def list_photos_api(self):
        """API-style photo listing."""
        return self.file_manager.list_all_photos()

    def get_photo_info_api(self, photo_id):
        """API-style photo info retrieval."""
        return self.file_manager.get_photo_info(photo_id)

    def delete_photo_api(self, photo_id):
        """API-style photo deletion."""
        return self.file_manager.delete_photo(photo_id)

    def reload_settings_api(self):
        """API-style settings reload."""
        return self.camera_manager.reload_settings()

    def apply_settings_api(self, camera_settings=None):
        """API-style settings application."""
        if camera_settings:
            self.camera_manager.apply_camera_settings(camera_settings)
        return {"success": True, "message": "Settings applied"}

    def get_system_status_api(self):
        """API-style system status."""
        import shutil

        try:
            # Get disk usage
            total, used, free = shutil.disk_usage(SAVE_PATH)

            # Count photos
            all_photos = self.list_photos_api()
            original_count = len(all_photos)
            dithered_count = sum(1 for p in all_photos if p["has_dithered"])

            return {
                "success": True,
                "storage": {
                    "total_gb": round(total / (1024**3), 2),
                    "used_gb": round(used / (1024**3), 2),
                    "free_gb": round(free / (1024**3), 2),
                    "usage_percent": round((used / total) * 100, 1)
                },
                "photos": {
                    "original_count": original_count,
                    "dithered_count": dithered_count,
                    "total_count": original_count
                },
                "camera_active": True,
                "display_active": True,
                "diagnostics": get_diagnostics(),
            }
        except Exception as e:
            logging.error(f"Error getting system status: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Could not get system status"
            }


# Shared camera system and operation lock for both button loop and API
camera_system: Optional[CameraSystem] = None
_operation_lock = threading.Lock()

# FastAPI application exposing hardware control over localhost (lazy initialized)
app = None  # Will be created when API server starts

def _create_fastapi_routes():
    """Create FastAPI app and routes when API server starts."""
    global app
    if not _lazy_import_fastapi():
        return None

    app = FastAPI(title="Reframe Hardware API")
    @app.post("/api/capture")
    def api_capture():
        global camera_system
        if camera_system is None:
            raise HTTPException(status_code=503, detail="Camera system not initialized")
        if camera_system.eink_display.is_busy():
            return {"success": False, "error": "display_busy", "message": "Display is refreshing, please wait"}
        with _operation_lock:
            camera_system.update_activity()
            result = camera_system.capture_photo_api()
        return result

    @app.post("/api/display/{photo_id}")
    def api_display(photo_id: str):
        global camera_system
        if camera_system is None:
            raise HTTPException(status_code=503, detail="Camera system not initialized")
        try:
            with _operation_lock:
                camera_system.update_activity()
                result = camera_system.display_photo_api(photo_id)
            return result
        except Exception as e:
            logging.error(f"Error displaying photo {photo_id}: {e}")
            return {"success": False, "error": str(e), "message": f"Failed to display photo: {str(e)}"}

    @app.post("/api/reprocess/{photo_id}")
    async def api_reprocess(photo_id: str, request: Request):
        global camera_system
        if camera_system is None:
            raise HTTPException(status_code=503, detail="Camera system not initialized")
        try:
            body: Dict[str, Any] = await request.json()
            processing_settings = body.get("processing_settings") if isinstance(body, dict) else None
        except Exception:
            processing_settings = None
        with _operation_lock:
            result = camera_system.reprocess_photo_api(photo_id, processing_settings)
        return result

    @app.get("/api/photos")
    def api_list_photos():
        global camera_system
        if camera_system is None:
            raise HTTPException(status_code=503, detail="Camera system not initialized")
        camera_system.update_activity()
        try:
            return camera_system.list_photos_api()
        except Exception as e:
            logging.warning(f"Error listing photos (may be mid-processing): {e}")
            return []  # Return empty list; dashboard will retry

    @app.get("/api/photos/{photo_id}")
    def api_get_photo(photo_id: str):
        global camera_system
        if camera_system is None:
            raise HTTPException(status_code=503, detail="Camera system not initialized")
        try:
            info = camera_system.get_photo_info_api(photo_id)
            if not info:
                raise HTTPException(status_code=404, detail="Photo not found")
            return info
        except HTTPException:
            raise
        except Exception as e:
            logging.warning(f"Error getting photo info for {photo_id}: {e}")
            raise HTTPException(status_code=404, detail="Photo not found or being processed")

    @app.delete("/api/photos/{photo_id}")
    def api_delete_photo(photo_id: str):
        global camera_system
        if camera_system is None:
            raise HTTPException(status_code=503, detail="Camera system not initialized")
        with _operation_lock:
            return camera_system.delete_photo_api(photo_id)

    @app.post("/api/settings/reload")
    def api_reload_settings():
        global camera_system
        if camera_system is None:
            raise HTTPException(status_code=503, detail="Camera system not initialized")
        with _operation_lock:
            result = camera_system.reload_settings_api()
            # Restart timeout monitor if settings changed
            camera_system.stop_timeout_monitor()
            camera_system.start_timeout_monitor()
            return result

    @app.post("/api/settings/apply")
    async def api_apply_settings(request: Request):
        global camera_system
        if camera_system is None:
            raise HTTPException(status_code=503, detail="Camera system not initialized")
        try:
            body: Dict[str, Any] = await request.json()
            camera_settings = body.get("camera_settings") if isinstance(body, dict) else None
        except Exception:
            camera_settings = None
        with _operation_lock:
            return camera_system.apply_settings_api(camera_settings)

    @app.get("/api/status")
    def api_status():
        global camera_system
        if camera_system is None:
            raise HTTPException(status_code=503, detail="Camera system not initialized")
        try:
            return camera_system.get_system_status_api()
        except Exception as e:
            logging.warning(f"Error getting system status: {e}")
            return {"success": False, "error": str(e), "message": "Could not get system status"}

    @app.get("/api/dashboard/access")
    def api_dashboard_access():
        global camera_system
        if camera_system is None:
            raise HTTPException(status_code=503, detail="Camera system not initialized")
        return camera_system.get_dashboard_access_api()

    @app.post("/api/dashboard/qr")
    def api_dashboard_qr():
        global camera_system
        if camera_system is None:
            raise HTTPException(status_code=503, detail="Camera system not initialized")
        with _operation_lock:
            return camera_system.display_dashboard_qr_api()

    @app.post("/api/timeout/reset")
    def api_reset_timeout():
        """Reset the timeout timer (extend the timeout period)."""
        global camera_system
        if camera_system is None:
            raise HTTPException(status_code=503, detail="Camera system not initialized")

        camera_system.update_activity()
        return {"status": "success", "message": "Timeout timer reset"}

    @app.get("/api/timeout/status")
    def api_get_timeout_status():
        """Get current timeout status and remaining time."""
        global camera_system
        if camera_system is None:
            raise HTTPException(status_code=503, detail="Camera system not initialized")

        timeout_enabled = camera_system.camera_manager.is_timeout_enabled()
        timeout_minutes = camera_system.camera_manager.get_timeout_minutes()

        if timeout_enabled:
            elapsed = camera_system.camera_manager.get_inactivity_seconds()
            remaining_seconds = max(0, (timeout_minutes * 60) - elapsed)
            remaining_minutes = remaining_seconds / 60
        else:
            remaining_seconds = None
            remaining_minutes = None

        return {
            "timeout_enabled": timeout_enabled,
            "timeout_minutes": timeout_minutes,
            "remaining_seconds": remaining_seconds,
            "remaining_minutes": remaining_minutes,
            "last_activity": time.time() - camera_system.camera_manager.get_inactivity_seconds()
        }

    @app.post("/api/display/clear")
    def api_clear_display():
        global camera_system
        if camera_system is None:
            raise HTTPException(status_code=503, detail="Camera system not initialized")
        with _operation_lock:
            return camera_system.eink_display.clear_display()

    return app


def _start_api_server_in_background(host: str = "127.0.0.1", port: int = 8077):

    app = _create_fastapi_routes()
    if app is None:
        logging.warning("FastAPI/uvicorn not available; hardware API will not be started")
        return
    def _run():
        try:
            uvicorn.run(app, host=host, port=port, log_level="warning")
        except Exception as e:
            logging.error(f"Failed to start API server: {e}")
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def main():
    global camera_system

    startup_settings_path = os.path.join(BASE_PATH, "settings.json")
    startup_display = EInkDisplay(startup_settings_path)
    if _auto_display_enabled(startup_settings_path):
        startup_display.prepare_async()

    # Python/Picamera2 imports happen before this point, overlapping the cold
    # boot wait for the camera subdevice. HDR is still applied before the
    # Picamera2 constructor opens the camera.
    _enable_camera_hdr()
    camera_system = CameraSystem(eink_display=startup_display)
    logging.info("Camera system initialized")
    logging.info("Taking startup photo...")
    startup_status = "Camera initialized; startup capture failed"
    try:
        with _operation_lock:
            result = camera_system.capture_photo_api(fast_mode=True)

        if result.get("success"):
            logging.info("Startup photo captured: %s", result.get("photo_id", "unknown"))
            if camera_system.camera_manager.settings.get("display", {}).get("auto_display", True):
                logging.info("Startup photo sent to display")
            logging.info("System ready")
            startup_status = "Startup photo dispatched"

            # Ensure activity time is updated before starting timeout monitor
            camera_system.update_activity()
            camera_system.start_timeout_monitor_deferred()
        else:
            logging.warning("Failed to capture startup photo: %s", result.get("message", "unknown error"))
    except Exception as e:
        logging.error("Error taking startup photo: %s", e)
    finally:
        # reframe.service is Type=notify. Dashboard startup waits for this, but
        # does not wait for the e-ink panel's long physical refresh.
        _notify_systemd_ready(startup_status)

    # ═══════════════════════════════════════════════════════════════
    # HARDWARE: Button — PiSugar 3 via I2C
    # The PiSugar 3 exposes a button register at I2C address 0x57.
    # This is a direct hardware read (no PiSugar library needed).
    # To use a different button/trigger, replace is_power_button_pressed().
    # Long press (≥2s) is ignored here — PiSugar handles shutdown natively.
    # See: https://github.com/PiSugar/PiSugar/wiki/PiSugar-3-I2C-Datasheet
    # ═══════════════════════════════════════════════════════════════
    I2C_ADDRESS = 0x57
    BUTTON_REGISTER = 0x02
    import smbus2
    bus = smbus2.SMBus(1)
    button_read_healthy = None
    last_button_error_log = 0.0

    def is_power_button_pressed():
        nonlocal button_read_healthy, last_button_error_log
        try:
            reg_val = bus.read_byte_data(I2C_ADDRESS, BUTTON_REGISTER)
            if button_read_healthy is not True:
                if button_read_healthy is False:
                    logging.info("Shutter button I2C connection recovered")
                    update_diagnostics(
                        "button",
                        activity="Shutter button connection recovered",
                    )
            button_read_healthy = True
            return bool(reg_val & 0x01)  # Check the least significant bit
        except Exception as e:
            now = time.monotonic()
            if button_read_healthy is not False:
                update_diagnostics(
                    "button",
                    "Shutter button is not responding. Check the PiSugar connection and I2C.",
                    "Shutter button stopped responding",
                )
            if button_read_healthy is not False or now - last_button_error_log >= 60:
                logging.error("Failed to read shutter button over I2C: %s", e)
                last_button_error_log = now
            button_read_healthy = False
            return False

    prev_state = False
    button_press_start_time = None
    LONG_PRESS_THRESHOLD = 2.0  # 2 seconds threshold for long press

    # Start API server in background
    try:
        port = int(os.environ.get("REFRAME_API_PORT", "8077"))
    except Exception:
        port = 8077
    _start_api_server_in_background(host="127.0.0.1", port=port)
    camera_system.start_dashboard_qr_monitor()

    logging.info("System initialized. API server running. Waiting for button press to capture photo...")
    logging.info(f"Button protection: Long press (>={LONG_PRESS_THRESHOLD}s) will not trigger photo capture")

    try:
        while True:
            current_state = is_power_button_pressed()

            # Button press started
            if current_state and not prev_state:
                button_press_start_time = time.monotonic()
                logging.info("Button pressed - monitoring for long press protection...")
                update_diagnostics(activity="Button press detected")

            # Button released
            elif not current_state and prev_state:
                if button_press_start_time is not None:
                    press_duration = time.monotonic() - button_press_start_time

                    if press_duration < LONG_PRESS_THRESHOLD:
                        # Block captures while display is mid-refresh to avoid
                        # invisible captures with no visual feedback
                        if camera_system.eink_display.is_busy():
                            logging.info(f"Short press detected ({press_duration:.1f}s) - display busy, ignoring")
                            update_diagnostics(
                                activity="Button press ignored while the display was refreshing",
                            )
                        else:
                            logging.info(f"Short press detected ({press_duration:.1f}s) - capturing photo...")
                            with _operation_lock:
                                result = camera_system.capture_photo_api()
                            if result.get("success"):
                                logging.info("Photo captured%s.", " and sent to display" if camera_system.camera_manager.settings.get("display", {}).get("auto_display", True) else "")
                            else:
                                logging.error("Capture failed: %s", result.get("message", "unknown error"))
                    else:
                        logging.info(f"Long press detected ({press_duration:.1f}s) - ignoring for photo capture")
                        update_diagnostics(
                            activity="Long press detected; no photo capture was requested",
                        )

                    button_press_start_time = None

            prev_state = current_state
            sleep(BUTTON_POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        logging.info("Program interrupted by user. Exiting...")
    finally:
        bus.close()
        try:
            # Stop timeout monitor and put display to sleep if initialized inside CameraSystem
            if camera_system:
                camera_system.stop_timeout_monitor()
                camera_system.stop_dashboard_qr_monitor()
                if camera_system.eink_display:
                    camera_system.eink_display.sleep()
        except Exception:
            pass


def demo_api_usage():
    """Demonstrate the new API-style functionality."""
    print("Initializing Camera System...")
    camera_system = CameraSystem()

    print("\n=== System Status ===")
    status = camera_system.get_system_status_api()
    print(f"Storage: {status['storage']['free_gb']}GB free")
    print(f"Photos: {status['photos']['total_count']} total")

    print("\n=== Capturing Photo ===")
    capture_result = camera_system.capture_photo_api()
    if capture_result["success"]:
        photo_id = capture_result["photo_id"]
        print(f"Captured photo: {photo_id}")

        print("\n=== Reprocessing with Different Settings ===")
        new_settings = {
            "dithering_method": "ordered",
            "bayer_size": 8,
            "saturation": 0.8
        }
        reprocess_result = camera_system.reprocess_photo_api(photo_id, new_settings)
        print(f"Reprocessing result: {reprocess_result['success']}")

        print("\n=== Displaying Photo ===")
        display_result = camera_system.display_photo_api(photo_id)
        print(f"Display result: {display_result['success']}")

    print("\n=== Listing All Photos ===")
    photos = camera_system.list_photos_api()
    print(f"Found {len(photos)} photos")
    for photo in photos[:3]:  # Show first 3
        print(f"  {photo['id']}: {photo['filename']} ({'dithered' if photo['has_dithered'] else 'original only'})")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Run the API demo
        try:
            demo_api_usage()
        except KeyboardInterrupt:
            print("\nDemo interrupted. Exiting...")
        except Exception as e:
            print(f"Demo error: {e}")
    else:
        # Run the original main loop
        try:
            main()
        except KeyboardInterrupt:
            logging.info("Program interrupted. Exiting...")
            epd4in0e.epdconfig.module_exit(cleanup=True)
        except Exception as error:
            logging.exception("Camera service stopped unexpectedly")
            record_service_failure(error)
            raise
