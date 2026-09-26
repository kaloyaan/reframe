#!/usr/bin/env python3

import os
import json
import sys
import logging
import asyncio
import time
import shutil
import math
import tempfile
import threading
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response
from starlette.background import BackgroundTask
import httpx
from PIL import Image

CAMERA_AVAILABLE = False

# Constants
BASE_PATH = os.path.dirname(os.path.realpath(__file__))
PHOTOS_PATH = os.path.join(BASE_PATH, "photos")
DITHERED_PHOTOS_PATH = os.path.join(BASE_PATH, "dithered_photos")
SETTINGS_PATH = os.path.join(BASE_PATH, "settings.json")
USER_DATA_PATHS = ["settings.json", "photos", "dithered_photos"]
UPDATE_HELPER_PATH = "/usr/local/sbin/reframe-apply-update"
UPDATE_PENDING_PATH = os.path.join(BASE_PATH, ".runtime", "update_pending")
DIAGNOSTICS_PATH = os.path.join(BASE_PATH, ".runtime", "diagnostics.json")
REFRAME_LOG_PATH = os.path.join(BASE_PATH, ".runtime", "reframe.log")

os.makedirs(PHOTOS_PATH, exist_ok=True)
os.makedirs(DITHERED_PHOTOS_PATH, exist_ok=True)

app = FastAPI(title="Reframe Dashboard", description="Control & Gallery Interface for Reframe Camera")


def load_diagnostics() -> Dict[str, Any]:
    """Read the last diagnostic snapshot when the camera service is down."""
    try:
        with open(DIAGNOSTICS_PATH, "r", encoding="utf-8") as diagnostic_file:
            record = json.load(diagnostic_file)
        return record if isinstance(record, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def tail_reframe_log(max_lines=100) -> str:
    """Return a bounded tail from the camera service's rotating log."""
    lines = []
    for path in (f"{REFRAME_LOG_PATH}.1", REFRAME_LOG_PATH):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as log_file:
                lines.extend(log_file.readlines())
        except OSError:
            continue
    return "".join(lines[-max_lines:])


def build_health_report(
    hardware_status: Optional[Dict[str, Any]],
    stored_diagnostics: Dict[str, Any],
    storage: Dict[str, int],
    battery: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the small status model used by the home screen and Health panel."""
    live = isinstance(hardware_status, dict) and hardware_status.get("success") is True
    diagnostics = hardware_status.get("diagnostics", {}) if live else stored_diagnostics
    issues = diagnostics.get("issues", {}) if isinstance(diagnostics, dict) else {}
    messages = {}
    for component, issue in issues.items():
        message = issue.get("message") if isinstance(issue, dict) else issue
        if isinstance(message, str) and message:
            messages[component] = message

    total_bytes = max(int(storage.get("total", 0)), 1)
    free_bytes = max(int(storage.get("free", 0)), 0)
    free_gb = free_bytes / (1024 ** 3)
    used_percent = ((total_bytes - free_bytes) / total_bytes) * 100
    if free_bytes < 500 * 1024 * 1024:
        messages.setdefault("storage", f"Only {free_gb:.1f} GB free; photos may not save")

    battery_level = battery.get("battery_level")
    if battery_level is not None:
        battery_level = int(battery_level)
        if battery_level <= 5:
            messages["battery"] = f"Battery critically low ({battery_level}%)"

    order = ("camera", "button", "display", "storage", "service", "battery")
    primary = next((
        component for component in order
        if component in messages or (component == "service" and not live)
    ), None)
    headlines = {
        "camera": "camera problem",
        "button": "button not responding",
        "display": "display problem",
        "storage": "storage full",
        "service": "camera service unavailable",
        "battery": "battery critically low",
    }
    headline = headlines.get(primary, "system issue")
    if primary == "camera" and "not detected" in messages["camera"].lower():
        headline = "camera not detected"
    last_activity = diagnostics.get("last_activity") if isinstance(diagnostics, dict) else None
    if isinstance(last_activity, dict):
        last_activity = last_activity.get("message")

    return {
        "system_online": primary is None,
        "headline": "system online" if primary is None else headline,
        "service": {"ok": live, "message": "Running" if live else "Not responding"},
        "issues": [messages[key] for key in order if key in messages],
        "storage": f"{free_gb:.1f} GB free ({used_percent:.0f}% used)",
        "battery": battery_level,
        "last_activity": last_activity,
    }


def prepare_dithered_export(image_path: str, upscale_2x: bool) -> tuple[bytes, str, str]:
    """Read a dithered image, optionally returning a lossless 2x PNG export."""
    source_path = Path(image_path)
    if not upscale_2x:
        return source_path.read_bytes(), source_path.name, "image/png"

    with Image.open(source_path) as image:
        resampling = getattr(Image, "Resampling", Image)
        enlarged = image.resize(
            (image.width * 2, image.height * 2),
            resampling.NEAREST
        )
        output = BytesIO()
        enlarged.save(output, format="PNG")

    return output.getvalue(), f"{source_path.stem}.png", "image/png"


class SettingsValidationError(ValueError):
    pass


def validate_settings(settings: Dict[str, Any]) -> None:
    """Validate persisted settings against the hardware and dashboard contract."""
    if not isinstance(settings, dict):
        raise SettingsValidationError("Settings must be a JSON object")

    def section(parent: Dict[str, Any], key: str, path: str) -> Dict[str, Any]:
        value = parent.get(key, {})
        if not isinstance(value, dict):
            raise SettingsValidationError(f"{path} must be an object")
        return value

    def number(value: Any, path: str, minimum: float, maximum: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SettingsValidationError(f"{path} must be a number")
        if value < minimum or value > maximum:
            raise SettingsValidationError(f"{path} must be between {minimum} and {maximum}")

    def integer(value: Any, path: str, minimum: int, maximum: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SettingsValidationError(f"{path} must be an integer")
        if value < minimum or value > maximum:
            raise SettingsValidationError(f"{path} must be between {minimum} and {maximum}")

    def boolean(value: Any, path: str) -> None:
        if not isinstance(value, bool):
            raise SettingsValidationError(f"{path} must be true or false")

    def text(value: Any, path: str, maximum: int, allow_controls: bool = False) -> None:
        if not isinstance(value, str):
            raise SettingsValidationError(f"{path} must be text")
        if len(value) > maximum:
            raise SettingsValidationError(f"{path} must be {maximum} characters or fewer")
        if not allow_controls and any(ord(char) < 32 for char in value):
            raise SettingsValidationError(f"{path} cannot contain control characters")

    camera = section(settings, "camera", "camera")
    resolution = section(camera, "resolution", "camera.resolution")
    integer(resolution.get("width"), "camera.resolution.width", 100, 4000)
    integer(resolution.get("height"), "camera.resolution.height", 100, 4000)
    number(camera.get("exposure_value"), "camera.exposure_value", -2, 2)
    number(camera.get("sharpness"), "camera.sharpness", 0, 10)
    if camera.get("autofocus_mode") not in {0, 1, 2}:
        raise SettingsValidationError("camera.autofocus_mode must be 0, 1, or 2")

    processing = section(settings, "processing", "processing")
    number(processing.get("saturation"), "processing.saturation", 0, 2)
    number(processing.get("brightness_factor"), "processing.brightness_factor", 0.1, 3)
    number(processing.get("color_factor"), "processing.color_factor", 0.1, 3)
    if processing.get("dithering_method") not in {"floyd_steinberg", "ordered"}:
        raise SettingsValidationError("processing.dithering_method is unsupported")
    if processing.get("bayer_size") not in {2, 4, 8}:
        raise SettingsValidationError("processing.bayer_size must be 2, 4, or 8")
    number(processing.get("threshold_scale"), "processing.threshold_scale", 0.1, 2)

    display = section(settings, "display", "display")
    boolean(display.get("auto_display"), "display.auto_display")
    number(display.get("display_timeout"), "display.display_timeout", 0, 3600)
    boolean(display.get("fast_refresh", False), "display.fast_refresh")

    system = section(settings, "system", "system")
    integer(system.get("auto_refresh_interval"), "system.auto_refresh_interval", 5, 300)
    integer(system.get("auto_timeout_minutes"), "system.auto_timeout_minutes", 1, 60)
    boolean(system.get("auto_timeout_enabled"), "system.auto_timeout_enabled")
    boolean(system.get("show_dashboard_qr_on_wifi_connect"), "system.show_dashboard_qr_on_wifi_connect")
    text(system.get("camera_name", ""), "system.camera_name", 80)

    exports = section(settings, "exports", "exports")
    boolean(exports.get("upscale_dithered_2x"), "exports.upscale_dithered_2x")

    extensions = section(settings, "extensions", "extensions")
    arena = section(extensions, "arena", "extensions.arena")
    boolean(arena.get("enabled"), "extensions.arena.enabled")
    text(arena.get("channel", ""), "extensions.arena.channel", 200)
    text(arena.get("access_token", ""), "extensions.arena.access_token", 4096)

class SettingsManager:
    """Manages settings operations for the dashboard."""
    
    def __init__(self, settings_path: str = SETTINGS_PATH):
        self.settings_path = Path(settings_path)
        self.default_settings = {
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
                "show_dashboard_qr_on_wifi_connect": True,
                "camera_name": ""
            },
            "exports": {
                "upscale_dithered_2x": False
            },
            "extensions": {
                "arena": {
                    "enabled": False,
                    "channel": "",
                    "access_token": ""
                }
            }
        }
        self._ensure_settings_file()
    
    def _ensure_settings_file(self):
        """Ensure settings file exists with default values."""
        if not self.settings_path.exists():
            self.save_settings(self.default_settings)
    
    def load_settings(self) -> Dict[str, Any]:
        """Load settings from JSON file."""
        try:
            with open(self.settings_path, 'r') as f:
                settings = json.load(f)
            # Ensure all default keys exist
            return self._merge_with_defaults(settings)
        except (FileNotFoundError, json.JSONDecodeError):
            return self.default_settings.copy()

    def load_public_settings(self) -> Dict[str, Any]:
        """Load settings safe to return to the browser."""
        settings = self.load_settings()
        arena_settings = settings.get("extensions", {}).get("arena", {})
        access_token = arena_settings.get("access_token", "")
        arena_settings["access_token"] = ""
        arena_settings["access_token_configured"] = bool(access_token)
        return settings
    
    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """Save settings to JSON file."""
        try:
            if not isinstance(settings, dict):
                raise SettingsValidationError("Settings must be a JSON object")
            # Merge with existing settings to preserve structure
            current_settings = self.load_settings()
            settings = self._prepare_settings_for_save(current_settings, settings)
            merged_settings = self._deep_merge(current_settings, settings)
            validate_settings(merged_settings)
            self._write_settings(merged_settings)
            return True
        except SettingsValidationError:
            raise
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False

    def replace_settings(self, settings: Dict[str, Any]) -> None:
        """Atomically replace settings with a previously validated snapshot."""
        validate_settings(settings)
        self._write_settings(settings)

    def _write_settings(self, settings: Dict[str, Any]) -> None:
        """Write JSON beside the target and atomically move it into place."""
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            prefix=f".{self.settings_path.name}.",
            dir=str(self.settings_path.parent)
        )
        try:
            with os.fdopen(fd, "w") as temp_file:
                json.dump(settings, temp_file, indent=2)
                temp_file.write("\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, self.settings_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _merge_with_defaults(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Merge loaded settings with defaults to ensure all keys exist."""
        system_settings = settings.get("system")
        if isinstance(system_settings, dict):
            legacy_value = system_settings.pop("show_dashboard_qr_on_first_network", None)
            if "show_dashboard_qr_on_wifi_connect" not in system_settings and legacy_value is not None:
                system_settings["show_dashboard_qr_on_wifi_connect"] = legacy_value
        return self._deep_merge(self.default_settings.copy(), settings)
    
    def _deep_merge(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _prepare_settings_for_save(self, current_settings: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
        """Apply write-only extension secret semantics before merging settings."""
        extensions = settings.get("extensions")
        if not isinstance(extensions, dict):
            return settings

        arena_settings = extensions.get("arena")
        if not isinstance(arena_settings, dict):
            return settings

        current_token = (
            current_settings
            .get("extensions", {})
            .get("arena", {})
            .get("access_token", "")
        )
        incoming_token = arena_settings.get("access_token")
        clear_token = arena_settings.pop("access_token_clear", False)
        if not isinstance(clear_token, bool):
            raise SettingsValidationError("extensions.arena.access_token_clear must be true or false")

        if clear_token:
            arena_settings["access_token"] = ""
        elif incoming_token is None or incoming_token == "":
            arena_settings["access_token"] = current_token

        arena_settings.pop("access_token_configured", None)
        return settings

    def clear_extension_secret(self, extension_id: str, secret_key: str) -> bool:
        """Clear one stored extension secret."""
        settings = self.load_settings()
        extension_settings = settings.get("extensions", {}).get(extension_id)
        if not isinstance(extension_settings, dict) or secret_key not in extension_settings:
            return False
        extension_settings[secret_key] = ""
        try:
            validate_settings(settings)
            self._write_settings(settings)
            return True
        except Exception as e:
            print(f"Error clearing extension secret: {e}")
            return False
    
    def get_camera_settings(self) -> Dict[str, Any]:
        """Get camera-specific settings."""
        return self.load_settings().get("camera", {})
    
    def get_processing_settings(self) -> Dict[str, Any]:
        """Get processing-specific settings."""
        return self.load_settings().get("processing", {})


class ReframeClient:
    """HTTP client to talk to the main reframe hardware service."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(30.0)

    async def get(self, path: str):
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()

    async def post(self, path: str, json: Optional[Dict[str, Any]] = None):
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=json)
            resp.raise_for_status()
            return resp.json()

    async def delete(self, path: str):
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.delete(url)
            resp.raise_for_status()
            return resp.json()


async def run_repo_command(args: List[str], timeout: int = 30) -> Dict[str, Any]:
    """Run a command in the repo and return captured output."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=BASE_PATH,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return {
            "returncode": 124,
            "stdout": "",
            "stderr": f"Command timed out: {' '.join(args)}"
        }
    except Exception as e:
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": str(e)
        }

    return {
        "returncode": proc.returncode,
        "stdout": stdout.decode("utf-8", errors="replace").strip(),
        "stderr": stderr.decode("utf-8", errors="replace").strip()
    }


async def run_git(args: List[str], timeout: int = 30) -> Dict[str, Any]:
    return await run_repo_command(["git", *args], timeout=timeout)


def git_error_detail(result: Dict[str, Any], fallback: str) -> str:
    return result.get("stderr") or result.get("stdout") or fallback


async def get_update_status(fetch: bool = True) -> Dict[str, Any]:
    """Fetch and compare this checkout with its upstream branch."""
    if not os.path.isdir(os.path.join(BASE_PATH, ".git")):
        raise HTTPException(status_code=501, detail="Software updates require a git checkout")

    if fetch:
        fetch_result = await run_git(["fetch", "--quiet", "origin"], timeout=60)
        if fetch_result["returncode"] != 0:
            detail = git_error_detail(fetch_result, "git fetch failed")
            raise HTTPException(status_code=502, detail=f"Could not check for updates: {detail}")

    branch_result = await run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    branch = branch_result["stdout"] if branch_result["returncode"] == 0 else "unknown"

    upstream_result = await run_git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    upstream = upstream_result["stdout"] if upstream_result["returncode"] == 0 else "origin/main"

    local_result = await run_git(["rev-parse", "HEAD"])
    remote_result = await run_git(["rev-parse", upstream])
    if local_result["returncode"] != 0 or remote_result["returncode"] != 0:
        raise HTTPException(status_code=500, detail="Could not determine current software version")

    local_revision = local_result["stdout"]
    remote_revision = remote_result["stdout"]
    base_result = await run_git(["merge-base", "HEAD", upstream])
    merge_base = base_result["stdout"] if base_result["returncode"] == 0 else ""

    counts_result = await run_git(["rev-list", "--left-right", "--count", f"HEAD...{upstream}"])
    ahead = 0
    behind = 0
    if counts_result["returncode"] == 0:
        parts = counts_result["stdout"].split()
        if len(parts) == 2:
            ahead = int(parts[0])
            behind = int(parts[1])

    dirty_result = await run_git(["status", "--porcelain", "--untracked-files=no"])
    has_tracked_changes = bool(dirty_result["stdout"]) if dirty_result["returncode"] == 0 else True

    update_available = local_revision != remote_revision and merge_base == local_revision
    up_to_date = local_revision == remote_revision
    diverged = local_revision != remote_revision and merge_base != local_revision
    post_install_pending = os.path.exists(UPDATE_PENDING_PATH)
    can_update = (update_available or post_install_pending) and not has_tracked_changes

    if post_install_pending:
        message = "Code is up to date, but installation steps still need to finish."
    elif up_to_date:
        message = "Software is up to date."
    elif update_available:
        message = f"Update available: {behind} commit{'s' if behind != 1 else ''} behind {upstream}."
    elif diverged:
        message = "This checkout has diverged from upstream and cannot be updated automatically."
    else:
        message = "Could not determine update state."

    if update_available and has_tracked_changes:
        message = f"{message} Local code changes must be committed or discarded before updating."

    return {
        "branch": branch,
        "upstream": upstream,
        "local_revision": local_revision,
        "remote_revision": remote_revision,
        "local_short": local_revision[:7],
        "remote_short": remote_revision[:7],
        "ahead": ahead,
        "behind": behind,
        "update_available": update_available,
        "up_to_date": up_to_date,
        "diverged": diverged,
        "has_tracked_changes": has_tracked_changes,
        "post_install_pending": post_install_pending,
        "can_update": can_update,
        "message": message
    }


def backup_settings_for_update() -> Optional[str]:
    """Back up settings before pulling code; ignored photo dirs are left untouched."""
    settings_path = Path(SETTINGS_PATH)
    if not settings_path.exists():
        return None

    backup_dir = Path(BASE_PATH) / ".update_backups" / datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(settings_path, backup_dir / "settings.json")
    return str(backup_dir)


def ignored_user_data_paths() -> List[str]:
    return [path for path in USER_DATA_PATHS if os.path.exists(os.path.join(BASE_PATH, path))]


class PhotoManager:
    """Manages photo operations for the dashboard."""
    
    def __init__(self):
        self.photos_path = Path(PHOTOS_PATH)
        self.dithered_path = Path(DITHERED_PHOTOS_PATH)
        
    def get_all_photos(self, page: int = 1, limit: int = 20) -> Dict[str, Any]:
        """Get paginated list of photos with metadata."""
        all_photos = []
        
        # Get all files from photos directory
        if self.photos_path.exists():
            for photo_file in sorted(self.photos_path.iterdir(), reverse=True):
                if photo_file.is_file() and photo_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Look for corresponding dithered version
                    dithered_file = self.dithered_path / f"{photo_file.stem}_dithered.png"
                    if not dithered_file.exists():
                        dithered_file = self.dithered_path / f"{photo_file.stem}_dithered{photo_file.suffix}"
                    if not dithered_file.exists():
                        # Try without _dithered suffix for exact matches
                        dithered_file = self.dithered_path / photo_file.name
                    
                    photo_info = {
                        "id": photo_file.stem,
                        "filename": photo_file.name,
                        "original_path": f"/photos/{photo_file.name}",
                        "dithered_path": f"/dithered/{dithered_file.name}" if dithered_file.exists() else None,
                        "has_dithered": dithered_file.exists(),
                        "size": photo_file.stat().st_size,
                        "created": datetime.fromtimestamp(photo_file.stat().st_mtime).isoformat()
                    }
                    all_photos.append(photo_info)
        
        # Calculate pagination
        total_photos = len(all_photos)
        total_pages = (total_photos + limit - 1) // limit  # Ceiling division
        start_index = (page - 1) * limit
        end_index = start_index + limit
        
        # Get photos for current page
        photos_page = all_photos[start_index:end_index]
        
        return {
            "photos": photos_page,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "total_photos": total_photos,
                "photos_per_page": limit,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        }
    
    def get_photo_info(self, photo_id: str) -> Dict:
        """Get information about a specific photo."""
        # Get all photos without pagination to search through them
        all_photos_data = self.get_all_photos(page=1, limit=10000)  # Large limit to get all
        photos = all_photos_data["photos"]
        for photo in photos:
            if photo["id"] == photo_id:
                return photo
        raise HTTPException(status_code=404, detail="Photo not found")


class DashboardExtension:
    """Base interface for dashboard photo extensions."""

    id = ""
    label = ""
    action_label = ""
    requires_dithered = True

    def get_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        return settings.get("extensions", {}).get(self.id, {})

    def enabled(self, settings: Dict[str, Any]) -> bool:
        return bool(self.get_settings(settings).get("enabled", False))

    def configured(self, settings: Dict[str, Any]) -> bool:
        return self.enabled(settings)

    def public_action(self, settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.configured(settings):
            return None
        return {
            "id": self.id,
            "label": self.label,
            "action_label": self.action_label,
            "requires_dithered": self.requires_dithered
        }

    async def run(self, photo: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class ArenaExtension(DashboardExtension):
    """Upload dithered photos to an Are.na channel using the v3 API."""

    id = "arena"
    label = "Are.na"
    action_label = "are.na"
    requires_dithered = True
    api_base = "https://api.are.na"
    s3_public_base = "https://s3.amazonaws.com/arena_images-temp"
    user_agent = "reFrame"
    max_retries = 3
    retry_delay = 2

    def configured(self, settings: Dict[str, Any]) -> bool:
        extension_settings = self.get_settings(settings)
        return (
            self.enabled(settings)
            and bool(extension_settings.get("channel"))
            and bool(extension_settings.get("access_token"))
        )

    async def run(self, photo: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
        extension_settings = self.get_settings(settings)
        channel = str(extension_settings.get("channel", "")).strip()
        access_token = str(extension_settings.get("access_token", "")).strip()
        dithered_path = photo.get("dithered_path")

        if not self.enabled(settings):
            raise HTTPException(status_code=400, detail="Are.na extension is disabled")
        if not channel:
            raise HTTPException(status_code=400, detail="Are.na channel is not configured")
        if not access_token:
            raise HTTPException(status_code=400, detail="Are.na access token is not configured")
        if not dithered_path:
            raise HTTPException(status_code=400, detail="This photo does not have a dithered version to upload")
        if not os.path.exists(dithered_path):
            raise HTTPException(status_code=404, detail="Dithered photo file was not found")

        upscale_2x = bool(settings.get("exports", {}).get("upscale_dithered_2x", False))
        try:
            upload_content, filename, upload_content_type = await asyncio.to_thread(
                prepare_dithered_export,
                dithered_path,
                upscale_2x
            )
        except OSError as error:
            raise HTTPException(status_code=500, detail=f"Could not prepare dithered upload: {error}") from error

        headers = {
            "Authorization": f"Bearer {access_token}",
            "User-Agent": self.user_agent
        }
        timeout = httpx.Timeout(60.0)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                presign_resp = await self._request_with_retry(
                    client,
                    "post",
                    f"{self.api_base}/v3/uploads/presign",
                    headers=headers,
                    json={"files": [{"filename": filename, "content_type": upload_content_type}]}
                )
                self._raise_for_arena_error(presign_resp, "Could not create Are.na upload URL")
                presign_data = presign_resp.json()
                presigned_file = (presign_data.get("files") or [None])[0]
                if not presigned_file:
                    raise HTTPException(status_code=502, detail="Are.na did not return an upload URL")

                upload_url = presigned_file.get("upload_url")
                key = presigned_file.get("key")
                content_type = presigned_file.get("content_type", upload_content_type)
                if not upload_url or not key:
                    raise HTTPException(status_code=502, detail="Are.na upload URL response was incomplete")

                upload_resp = await client.put(
                    upload_url,
                    content=upload_content,
                    headers={"Content-Type": content_type}
                )
                if upload_resp.status_code >= 400:
                    raise HTTPException(status_code=502, detail="Upload to Are.na storage failed")

                s3_url = f"{self.s3_public_base}/{key}"
                photo_id = photo.get("id", "unknown")
                created = photo.get("created_at") or photo.get("created")
                description = self._build_block_description(created, settings)

                block_resp = await self._request_with_retry(
                    client,
                    "post",
                    f"{self.api_base}/v3/blocks",
                    headers=headers,
                    json={
                        "value": s3_url,
                        "channels": [{"id": channel}],
                        "title": f"reFrame {photo_id}",
                        "description": description,
                        "metadata": {
                            "source": "reframe",
                            "photo_id": photo_id
                        }
                    }
                )
                self._raise_for_arena_error(block_resp, "Could not create Are.na block")
                block = block_resp.json()
        except HTTPException:
            raise
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Could not reach Are.na: {e}") from e

        block_id = block.get("id")
        block_url = (
            block.get("url")
            or block.get("href")
            or block.get("_links", {}).get("self", {}).get("href")
        )
        return {
            "status": "success",
            "message": "Uploaded dithered photo to Are.na",
            "extension": self.id,
            "block_id": block_id,
            "url": block_url
        }

    async def _request_with_retry(self, client, method: str, url: str, **kwargs) -> httpx.Response:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await getattr(client, method)(url, **kwargs)
            except httpx.RequestError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                raise

            if response.status_code == 429 and attempt < self.max_retries - 1:
                await asyncio.sleep(self._rate_limit_wait_seconds(response))
                continue

            if response.status_code >= 500 and attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)
                continue

            return response

        if last_error:
            raise last_error
        raise HTTPException(status_code=502, detail="Are.na request failed after retries")

    def _rate_limit_wait_seconds(self, response: httpx.Response) -> int:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(1, min(60, int(float(retry_after))))
            except ValueError:
                pass

        reset = response.headers.get("X-RateLimit-Reset")
        if reset:
            try:
                return max(1, min(60, int(float(reset)) - int(time.time())))
            except ValueError:
                pass

        return self.retry_delay

    def _raise_for_arena_error(self, response: httpx.Response, fallback: str) -> None:
        if response.status_code < 400:
            return

        detail_by_status = {
            401: "Are.na access token is invalid or missing",
            403: "Are.na token does not have write access or cannot post to this channel",
            404: "Are.na channel was not found",
            408: "Are.na request timed out",
            429: "Are.na rate limit reached; try again later"
        }
        detail = detail_by_status.get(response.status_code, fallback)

        try:
            data = response.json()
            message = data.get("details", {}).get("message") or data.get("message")
            if message:
                detail = f"{detail}: {message}"
        except Exception:
            pass

        status_code = response.status_code if response.status_code in detail_by_status else 502
        raise HTTPException(status_code=status_code, detail=detail)

    def _build_block_description(self, created, settings: Dict[str, Any]) -> str:
        description = "Dithered photo shot on [reframe.camera](https://reframe.camera)"
        captured = self._format_captured_at(created)
        if captured:
            camera_name = str(settings.get("system", {}).get("camera_name", "")).strip()
            attribution = f" by {camera_name}" if camera_name else ""
            description = f"{description}\n\nCaptured on {captured}{attribution}"
        return description

    def _format_captured_at(self, created) -> Optional[str]:
        if not created:
            return None

        try:
            if isinstance(created, (int, float)):
                dt = datetime.fromtimestamp(created, timezone.utc)
            elif isinstance(created, str):
                normalized = created.replace("Z", "+00:00")
                dt = datetime.fromisoformat(normalized)
            else:
                return str(created)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone.utc)
            return f'{dt.strftime("%B %-d, %Y at %-I:%M %p")} UTC'
        except Exception:
            return str(created)


class ExtensionRegistry:
    """Registry for server-side dashboard extensions."""

    def __init__(self, extensions: List[DashboardExtension]):
        self.extensions = {extension.id: extension for extension in extensions}

    def get(self, extension_id: str) -> Optional[DashboardExtension]:
        return self.extensions.get(extension_id)

    def public_actions(self, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
        actions = []
        for extension in self.extensions.values():
            action = extension.public_action(settings)
            if action:
                actions.append(action)
        return actions

# Initialize managers
settings_manager = SettingsManager()
photo_manager = PhotoManager()
extension_registry = ExtensionRegistry([ArenaExtension()])

# Initialize HTTP client to the hardware service
REFRAME_API_BASE = os.environ.get("REFRAME_API_BASE", "http://127.0.0.1:8077/api")
reframe_client = ReframeClient(REFRAME_API_BASE)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard interface."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
        <title>Reframe</title>
        <style>
            :root {
                --primary-color: #F5F1F0;  /* background */
                --secondary-color: #181818;             /* text and borders */
                --panel-border-color: #242424;           /* dotted panel outlines */
                --tertiary-color: #F5F1F0;              /* content backgrounds */
                --hover-color: #333;                  /* hover state color */
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: serif;
                background: var(--primary-color);
                min-height: 100vh;
                color: var(--secondary-color);
            }

            html.settings-open,
            body.settings-open {
                overflow: hidden;
            }

            body.settings-open {
                position: fixed;
                width: 100%;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 40px;
            }
            
            .header {
                text-align: left;
                margin-bottom: 60px;
            }
            
            .header h1 {
                color: var(--secondary-color);
                font-size: 1rem;
                font-weight: normal;
            }
            
            .header p {
                color: var(--secondary-color);
                font-size: 1rem;
                opacity: 0.8;
            }
            
            .controls {
                background: var(--tertiary-color);
                padding: 40px;
                margin-bottom: 40px;
            }
            
            .button {
                background: var(--secondary-color);
                color: var(--tertiary-color);
                border: none;
                padding: 5px 10px;
                cursor: pointer;
                font-size: 1rem;
                font-family: serif;
            }
            
            .button:hover {
                background: var(--hover-color);
            }
            
            .gallery {
                /* background: white; */
                /* padding: 40px; */
            }
            
            .pagination {
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 10px;
                margin: 40px 0;
                padding: 20px;
            }
            
            .pagination-btn {
                background: var(--tertiary-color);
                color: var(--secondary-color);
                border: 1px solid var(--secondary-color);
                padding: 5px 10px;
                cursor: pointer;
                font-family: serif;
                font-size: 1rem;
                min-width: 40px;
                text-align: center;
            }
            
            .pagination-btn:hover:not(:disabled) {
                background: var(--secondary-color);
                color: var(--tertiary-color);
            }
            
            .pagination-btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            .pagination-btn.active {
                background: var(--secondary-color);
                color: var(--tertiary-color);
            }
            
            .pagination-info {
                font-family: serif;
                font-size: 1rem;
                margin: 0 20px;
            }

            .pagination-ellipsis {
                display: inline-flex;
                align-items: center;
                padding: 0 4px;
            }

            #page-numbers {
                display: flex;
                align-items: center;
                gap: 4px;
            }

            .pagination-jump {
                display: flex;
                align-items: center;
                gap: 6px;
                white-space: nowrap;
            }

            .pagination-jump input {
                width: 64px;
                height: 32px;
                border: 1px solid var(--secondary-color);
                background: var(--tertiary-color);
                color: var(--secondary-color);
                padding: 4px 6px;
                font-family: serif;
                font-size: 1rem;
                text-align: center;
            }

            .gallery h2 {
                margin-bottom: 40px;
                color: var(--secondary-color);
                border-bottom: 4px solid var(--secondary-color);
                padding-bottom: 20px;
                font-size: 1rem;
                font-weight: normal;
            }
            
            .photo-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
            }
            
            .photo-card {
                border: 1.5px dotted var(--panel-border-color);
                overflow: hidden;
                background: var(--tertiary-color);
                position: relative;
            }
                        
            .photo-image {
                width: 100%;
                object-fit: cover;
                cursor: pointer;
                display: block;
            }
            
            .photo-info {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                background: rgba(0, 0, 0, 0.7);
                opacity: 0;
                pointer-events: none;
                transition: opacity 0.3s ease;
                padding: 10px;
            }
            
            .photo-card:hover .photo-info {
                opacity: 1;
                pointer-events: auto;
            }
            
            .photo-card.active .photo-info {
                opacity: 1;
                pointer-events: auto;
            }
            
            @media (hover: none) and (pointer: coarse) {
                .photo-card:hover .photo-info {
                    opacity: 0;
                    pointer-events: none;
                }
                
                .photo-card.active .photo-info {
                    opacity: 1;
                    pointer-events: auto;
                }
            }
            
            .photo-actions {
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
                justify-content: center;
            }

            .action-btn {
                padding: 8px 12px;
                font-size: 0.9rem;
                text-decoration: none;
                font-family: serif;
                border: 1px solid var(--tertiary-color);
                background: rgba(255, 255, 255, 0.9);
                color: var(--secondary-color);
                transition: all 0.2s ease;
                display: inline-flex;
                align-items: center;
                backdrop-filter: blur(2px);
            }
            
            .btn-primary {
                background: rgba(255, 255, 255, 0.9);
                color: var(--secondary-color);
            }
            
            .btn-secondary {
                background: rgba(255, 255, 255, 0.9);
                color: var(--secondary-color);
            }
            
            .btn-success {
                background: rgba(255, 255, 255, 0.9);
                color: var(--secondary-color);
            }
            
            .action-btn:hover {
                background: var(--tertiary-color);
                color: var(--secondary-color);
                transform: translateY(-1px);
            }
            
            .loading {
                text-align: center;
                padding: 60px;
                color: var(--secondary-color);
                font-size: 1rem;
            }
            
            .status-bar {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .status-item {
                display: flex;
                align-items: center;
                gap: 15px;
                font-size: 1rem;
            }
            
            .status-indicator {
                width: 12px;
                height: 12px;
                background: var(--secondary-color);
            }

            .status-indicator.issue {
                background: #d32f2f;
            }
            
            .settings-modal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.8);
                overflow: hidden;
                overscroll-behavior: contain;
            }
            
            .settings-content {
                background-color: var(--tertiary-color);
                margin: 3vh auto;
                padding: 32px;
                border: none;
                width: min(94vw, 1200px);
                height: 94vh;
                box-sizing: border-box;
                overflow-y: auto;
                overscroll-behavior: contain;
                -webkit-overflow-scrolling: touch;
            }
            
            .settings-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 24px;
            }
            
            .settings-header h2 {
                font-size: 1rem;
                font-weight: normal;
            }

            .settings-header-actions {
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .settings-header-actions .save-btn {
                margin-right: 0;
                padding: 10px 20px;
            }
            
            .close-btn {
                background: var(--secondary-color);
                color: var(--tertiary-color);
                border: none;
                padding: 10px 20px;
                cursor: pointer;
                font-family: serif;
                font-size: 1rem;
            }
            
            .close-btn:hover {
                background: var(--hover-color);
            }

            .settings-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                grid-template-areas:
                    "health health"
                    "camera processing"
                    "system processing"
                    "system extensions"
                    "exports extensions"
                    "updates extensions";
                gap: 20px;
                align-items: stretch;
            }
            
            .settings-section {
                border: 1.5px dotted var(--panel-border-color);
                padding: 20px;
            }

            .settings-camera {
                grid-area: camera;
            }

            .settings-health {
                grid-area: health;
            }

            .settings-processing {
                grid-area: processing;
            }

            .settings-system {
                grid-area: system;
            }

            .settings-updates {
                grid-area: updates;
            }

            .settings-exports {
                grid-area: exports;
            }

            .settings-extensions {
                grid-area: extensions;
            }
            
            .settings-section h3 {
                font-size: 1rem;
                margin-bottom: 20px;
                font-weight: normal;
                border-bottom: 1px solid var(--secondary-color);
                padding-bottom: 10px;
            }

            .health-row {
                display: grid;
                grid-template-columns: minmax(120px, 0.35fr) 1fr;
                gap: 14px;
                padding: 10px 0;
                border-top: 1px dotted var(--panel-border-color);
            }

            .health-row:first-of-type {
                border-top: none;
            }

            .health-row strong {
                font-weight: bold;
                font-size: 0.9rem;
                color: #666;
            }

            .health-row .issue {
                color: #d32f2f;
            }

            .health-actions {
                display: flex;
                gap: 8px;
                margin-top: 14px;
            }

            .setting-group {
                margin-bottom: 20px;
                display: flex;
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }
            
            .setting-label {
                font-weight: bold;
                font-size: 0.9rem;
                color: #666;
            }
            
            .setting-input {
                border: 1px solid var(--secondary-color);
                padding: 8px 12px;
                font-family: serif;
                font-size: 1rem;
                min-width: 100px;
                width: 100%;
                height: 44px;
                border-radius: 0;
                background-color: var(--tertiary-color);
                color: var(--secondary-color);
            }

            select.setting-input {
                appearance: none;
                -webkit-appearance: none;
                padding-right: 38px;
                background-image:
                    linear-gradient(45deg, transparent 50%, var(--secondary-color) 50%),
                    linear-gradient(135deg, var(--secondary-color) 50%, transparent 50%);
                background-position:
                    calc(100% - 15px) 19px,
                    calc(100% - 10px) 19px;
                background-size: 5px 5px, 5px 5px;
                background-repeat: no-repeat;
            }
            
            .setting-input:focus {
                /* outline: 2px solid var(--secondary-color); */
            }
            
            .setting-group > div {
                display: flex;
                flex-direction: column;
                gap: 5px;
                width: 100%;
            }
            
            .setting-help {
                font-size: 0.9rem;
                color: #666;
                margin-top: 5px;
            }
            
            .disabled-setting {
                opacity: 0.5;
                pointer-events: none;
            }
            
            .disabled-setting .setting-label {
                color: #999;
            }
            
            .disabled-setting .setting-input {
                background-color: #f5f5f5;
                color: #999;
                cursor: not-allowed;
            }
            
            #bayer-settings, #threshold-settings {
                display: none;
            }
            
            .settings-footer-actions {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 12px;
                margin-top: 24px;
                align-items: start;
            }

            .settings-footer-actions > div {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }

            .settings-footer-actions button {
                width: 100%;
                min-width: 0;
                min-height: 48px;
                margin: 0;
                padding: 12px 16px;
            }

            .danger-btn {
                background: #d32f2f;
            }
            
            .save-btn {
                background: var(--secondary-color);
                color: var(--tertiary-color);
                border: none;
                padding: 15px 30px;
                cursor: pointer;
                font-family: serif;
                font-size: 1rem;
                margin-right: 20px;
            }
            
            .save-btn:hover {
                background: var(--hover-color);
            }
            
            .reset-btn {
                background: var(--tertiary-color);
                color: var(--secondary-color);
                border: 1px solid var(--secondary-color);
                padding: 15px 30px;
                cursor: pointer;
                font-family: serif;
                font-size: 1rem;
            }
            
            .reset-btn:hover {
                background: var(--secondary-color);
                color: var(--tertiary-color);
            }
            
            @media (max-width: 768px) {
                .settings-modal {
                    height: 100vh;
                    height: -webkit-fill-available;
                    height: 100dvh;
                }

                .photo-grid {
                    grid-template-columns: 1fr;
                }
                
                .header h1 {
                    font-size: 1rem;
                }
                
                .container {
                    padding: 20px;
                }
                
                .status-bar {
                    flex-direction: row;
                    flex-wrap: wrap;
                    gap: 20px;
                    text-align: center;
                }
                
                .settings-content {
                    margin: 8px auto;
                    padding: 20px;
                    padding-bottom: calc(28px + env(safe-area-inset-bottom, 0px));
                    width: 95%;
                    height: calc(100% - 16px);
                    scroll-padding-bottom: calc(28px + env(safe-area-inset-bottom, 0px));
                }

                .settings-header {
                    align-items: flex-start;
                    gap: 16px;
                }

                .settings-header-actions {
                    flex-wrap: wrap;
                    justify-content: flex-end;
                    gap: 8px;
                }

                .settings-header-actions .save-btn,
                .settings-header-actions .close-btn {
                    padding: 10px 12px;
                }

                .settings-grid {
                    grid-template-columns: 1fr;
                    grid-template-areas:
                        "health"
                        "camera"
                        "processing"
                        "system"
                        "exports"
                        "updates"
                        "extensions";
                }

                .settings-footer-actions {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                    padding-bottom: env(safe-area-inset-bottom, 0px);
                }
                
                .setting-group {
                    flex-direction: column;
                    align-items: flex-start;
                }
                
                .setting-label {
                    min-width: auto;
                }
                
                .setting-help {
                    margin-top: 5px;
                }
                
                .pagination {
                    flex-wrap: wrap;
                    gap: 5px;
                }
                
                .pagination-info {
                    margin: 10px 0;
                    width: 100%;
                    text-align: center;
                }

                #page-numbers {
                    flex-wrap: wrap;
                    justify-content: center;
                }

                .pagination-jump {
                    width: 100%;
                    justify-content: center;
                    margin-top: 8px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="status-bar">
                <h1>reframe.camera dashboard</h1>
                <div class="status-item">
                    <div class="status-indicator" id="system-status-indicator"></div>
                    <span id="system-status-text">system online</span>
                </div>
                <div class="status-item">
                    <span id="battery-level">battery: --%</span>
                </div>
                <div class="status-item">
                    <span id="photo-count">loading photos...</span>
                </div>
                <div class="status-item">
                    <button class="button" onclick="refreshGallery()">refresh</button>
                </div>
                <button class="button" onclick="capturePhoto()">capture photo</button>
                <button class="button" onclick="openSettings()">settings</button>
            </div>
            </div>
            
            <div class="gallery">
                <div id="photo-grid" class="photo-grid">
                    <div class="loading">loading photos...</div>
                </div>
                
                <div id="pagination" class="pagination" style="display: none;">
                    <button class="pagination-btn" id="prev-btn" onclick="changePage(currentPage - 1)">previous</button>
                    <div id="page-numbers"></div>
                    <div class="pagination-info">
                        <span id="pagination-info"></span>
                    </div>
                    <form class="pagination-jump" onsubmit="jumpToPage(event)">
                        <label for="page-jump-input">page</label>
                        <input type="number" id="page-jump-input" min="1" step="1" inputmode="numeric" aria-label="Go to page">
                        <button class="pagination-btn" type="submit">go</button>
                    </form>
                    <button class="pagination-btn" id="next-btn" onclick="changePage(currentPage + 1)">next</button>
                </div>
            </div>
        </div>
        
        <!-- Settings Modal -->
        <div id="settings-modal" class="settings-modal">
            <div class="settings-content">
                <div class="settings-header">
                    <h2>settings</h2>
                    <div class="settings-header-actions">
                        <button class="save-btn" onclick="saveSettings()">save settings</button>
                        <button class="close-btn" onclick="closeSettings()">close</button>
                    </div>
                </div>
                
                <div class="settings-grid">
                <div class="settings-section settings-health" id="health-section">
                    <h3>health</h3>
                    <div class="health-row">
                        <strong>camera service</strong>
                        <span id="health-service">checking...</span>
                    </div>
                    <div class="health-row">
                        <strong>current issues</strong>
                        <span id="health-issues">none</span>
                    </div>
                    <div class="health-row">
                        <strong>storage</strong>
                        <span id="health-storage">checking...</span>
                    </div>
                    <div class="health-row">
                        <strong>battery</strong>
                        <span id="health-battery">checking...</span>
                    </div>
                    <div class="health-row">
                        <strong>last activity</strong>
                        <span id="health-last-activity">none recorded</span>
                    </div>
                    <div class="health-actions">
                        <form action="/api/diagnostics/logs" method="get" target="_blank">
                            <button class="button" type="submit">show logs</button>
                        </form>
                    </div>
                </div>

                <div class="settings-section settings-camera">
                    <h3>camera settings</h3>
                    <div class="setting-group disabled-setting">
                        <div>
                            <span class="setting-label">resolution width</span>
                            <input type="number" id="resolution-width" class="setting-input" min="100" max="4000" disabled>
                        </div>
                    </div>
                    <div class="setting-group disabled-setting">
                        <div>
                            <span class="setting-label">resolution height</span>
                            <input type="number" id="resolution-height" class="setting-input" min="100" max="4000" disabled>
                        </div>
                    </div>
                    <div class="setting-group">
                        <div>
                            <span class="setting-label">exposure value</span>
                            <input type="number" id="exposure-value" class="setting-input" step="0.25" min="-2" max="2">
                        </div>
                    </div>
                    <div class="setting-group">
                        <div>
                            <span class="setting-label">sharpness</span>
                            <input type="number" id="sharpness" class="setting-input" min="0" max="10">
                        </div>
                    </div>
                    <div class="setting-group">
                        <div>
                            <span class="setting-label">autofocus mode</span>
                            <select id="autofocus-mode" class="setting-input">
                                <option value="0">Manual</option>
                                <option value="1">Auto</option>
                                <option value="2">Continuous</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="settings-section settings-processing">
                    <h3>processing settings</h3>
                    <div class="setting-group">
                        <div>
                            <span class="setting-label">saturation</span>
                            <input type="number" id="saturation" class="setting-input" step="0.1" min="0" max="2">
                        </div>
                        <div class="setting-help">Blends between muted and saturated 6-color palettes; recommended 0.55–0.70 for natural tones.</div>
                    </div>
                    <div class="setting-group">
                        <div>
                            <span class="setting-label">brightness factor</span>
                            <input type="number" id="brightness-factor" class="setting-input" step="0.1" min="0.1" max="3">
                        </div>
                        <div class="setting-help">Multiplies image brightness before dithering; recommended 1.0–1.1.</div>
                    </div>
                    <div class="setting-group">
                        <div>
                            <span class="setting-label">color factor</span>
                            <input type="number" id="color-factor" class="setting-input" step="0.1" min="0.1" max="3">
                        </div>
                        <div class="setting-help">Boosts color intensity before dithering; recommended 1.1–1.3.</div>
                    </div>
                    <div class="setting-group">
                        <div>
                            <span class="setting-label">dithering method</span>
                            <select id="dithering-method" class="setting-input">
                                <option value="floyd_steinberg">floyd steinberg</option>
                                <option value="ordered">ordered (bayer)</option>
                            </select>
                        </div>
                        <div class="setting-help">Floyd–Steinberg is the default; ordered is still experimental</div>
                    </div>
                    <div class="setting-group" id="bayer-settings">
                        <div>
                            <span class="setting-label">bayer matrix size</span>
                            <select id="bayer-size" class="setting-input">
                                <option value="2">2x2</option>
                                <option value="4">4x4</option>
                                <option value="8">8x8</option>
                            </select>
                        </div>
                    </div>
                    <div class="setting-group" id="threshold-settings">
                        <div>
                            <span class="setting-label">threshold scale</span>
                            <input type="number" id="threshold-scale" class="setting-input" min="0.1" max="2.0" step="0.1">
                        </div>
                    </div>
                </div>
                
                <div class="settings-section settings-system">
                    <h3>system settings</h3>
                    <div class="setting-group">
                        <div>
                            <span class="setting-label">camera name</span>
                            <input type="text" id="camera-name" class="setting-input" maxlength="80" placeholder="optional">
                        </div>
                        <div class="setting-help">Used in Are.na upload descriptions when set.</div>
                    </div>
                    <div class="setting-group">
                        <div>
                            <span class="setting-label">auto refresh interval (seconds)</span>
                            <input type="number" id="auto-refresh-interval" class="setting-input" min="5" max="300">
                        </div>
                    </div>
                    <div class="setting-group">
                        <div>
                            <span class="setting-label">auto timeout enabled</span>
                            <select id="auto-timeout-enabled" class="setting-input">
                                <option value="true">enabled</option>
                                <option value="false">disabled</option>
                            </select>
                        </div>
                        <div class="setting-help">⚠️ Automatically shuts down the entire Raspberry Pi after inactivity to save battery. You'll need to manually power on the device to use it again.</div>
                    </div>
                    <div class="setting-group">
                        <div>
                            <span class="setting-label">auto timeout duration (minutes)</span>
                            <input type="number" id="auto-timeout-minutes" class="setting-input" min="1" max="60">
                        </div>
                        <div class="setting-help">Time of inactivity before system shuts down automatically</div>
                    </div>
                    <div class="setting-group">
                        <div>
                            <span class="setting-label">dashboard QR on Wi-Fi connect</span>
                            <select id="show-dashboard-qr-on-wifi-connect" class="setting-input">
                                <option value="true">enabled</option>
                                <option value="false">disabled</option>
                            </select>
                        </div>
                        <div class="setting-help">Shows the numeric dashboard address whenever the camera connects or reconnects.</div>
                    </div>
                    <div class="setting-group">
                        <button class="button" onclick="showDashboardQr()" type="button">show dashboard QR</button>
                    </div>

                </div>

                <div class="settings-section settings-exports">
                    <h3>advanced</h3>
                    <div class="setting-group">
                        <div>
                            <label class="setting-label" for="fast-refresh">screen refresh</label>
                            <select id="fast-refresh" class="setting-input">
                                <option value="false">standard (~21 seconds)</option>
                                <option value="true">fast (~11 seconds, experimental)</option>
                            </select>
                        </div>
                        <div class="setting-help">Fast mode uses an undocumented setting to drive the ePaper display twice as fast. Long-term effects on the display are untested.</div>
                    </div>
                    <div class="setting-group">
                        <div>
                            <span class="setting-label">2× dithered downloads and uploads</span>
                            <select id="upscale-dithered-2x" class="setting-input">
                                <option value="false">disabled</option>
                                <option value="true">enabled</option>
                            </select>
                        </div>
                        <div class="setting-help">Doubles dithered PNG dimensions with nearest-neighbor scaling only when downloading one photo or uploading it to Are.na. This tends to help the dithered photos look sharp on social media. Stored photos and ZIP downloads are unchanged.</div>
                    </div>
                </div>

                <div class="settings-section settings-updates">
                    <h3>software updates</h3>
                    <div class="setting-group">
                        <button class="button" id="update-check-btn" onclick="checkForUpdates()" type="button">check for updates</button>
                        <button class="button" id="update-install-btn" onclick="installUpdate()" type="button" style="display: none;">install update</button>
                        <div class="setting-help" id="update-status">Updates code, dependencies, and service files while preserving settings and photos.</div>
                    </div>
                </div>

                <div class="settings-section settings-extensions">
                    <h3>extensions</h3>
                    <div class="setting-group">
                        <div>
                            <span class="setting-label">are.na upload</span>
                            <select id="arena-enabled" class="setting-input">
                                <option value="false">disabled</option>
                                <option value="true">enabled</option>
                            </select>
                        </div>
                    </div>
                    <div class="setting-group">
                        <div>
                            <span class="setting-label">are.na channel slug or id</span>
                            <input type="text" id="arena-channel" class="setting-input" placeholder="my-channel">
                        </div>
                    </div>
                    <div class="setting-group">
                        <div>
                            <span class="setting-label">are.na access token</span>
                            <input type="password" id="arena-access-token" class="setting-input" placeholder="leave blank to keep saved token">
                        </div>
                        <div class="setting-help" id="arena-token-status">No token configured</div>
                    </div>
                    <div class="setting-group">
                        <button class="reset-btn" id="arena-clear-token-btn" onclick="clearArenaToken()" type="button">clear are.na token</button>
                    </div>
                </div>
                </div>
                
                <div class="settings-footer-actions">
                    <button class="save-btn" onclick="saveSettings()">save settings</button>
                    <button class="reset-btn" onclick="resetSettings()">reset to defaults</button>
                    <div id="download-controls">
                        <button class="button" onclick="downloadAllPhotos()">download all photos</button>
                        <button class="button danger-btn" onclick="abortDownload()" id="abort-btn" style="display: none;">abort download</button>
                    </div>
                    <div id="delete-controls">
                        <button class="button danger-btn" onclick="deleteAllPhotos()">delete all photos</button>
                        <button class="button danger-btn" onclick="abortDelete()" id="abort-delete-btn" style="display: none;">abort deletion</button>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let photos = [];
            let pagination = {};
            const requestedInitialPage = parseInt(new URLSearchParams(window.location.search).get('page'), 10);
            let currentPage = Number.isInteger(requestedInitialPage) && requestedInitialPage > 0 ? requestedInitialPage : 1;
            let latestPhotoLoadRequest = 0;
            let extensionActions = [];
            let arenaTokenShouldClear = false;
            let settingsFormSnapshot = null;
            let settingsPageScrollY = 0;
            const photosPerPage = 12;  // Show 12 photos per page
            
            // Function to notify backend of user activity
            async function notifyUserActivity() {
                try {
                    await fetch('/api/timeout/reset', { method: 'POST' });
                } catch (error) {
                    console.log('Could not notify user activity:', error);
                }
            }

            async function updateHealth() {
                const indicator = document.getElementById('system-status-indicator');
                const statusText = document.getElementById('system-status-text');
                try {
                    const response = await fetch('/api/health', { cache: 'no-store' });
                    if (!response.ok) throw new Error(`Health API returned ${response.status}`);
                    const data = await response.json();

                    indicator.classList.toggle('issue', !data.system_online);
                    statusText.textContent = data.headline || 'system online';

                    const service = document.getElementById('health-service');
                    service.textContent = data.service.message;
                    service.classList.toggle('issue', !data.service.ok);

                    const issues = document.getElementById('health-issues');
                    issues.textContent = data.issues.length ? data.issues.join(' • ') : 'none';
                    issues.classList.toggle('issue', data.issues.length > 0);

                    document.getElementById('health-storage').textContent = data.storage;
                    document.getElementById('health-battery').textContent =
                        data.battery === null ? 'Unavailable' : `${data.battery}%`;

                    document.getElementById('health-last-activity').textContent =
                        data.last_activity || 'none recorded';

                    const batteryElement = document.getElementById('battery-level');
                    if (data.battery !== null) {
                        batteryElement.textContent = `battery: ${data.battery}%`;
                        batteryElement.style.color = data.battery <= 10 ? '#d32f2f' : '';
                    } else {
                        batteryElement.textContent = 'battery: --%';
                        batteryElement.style.color = '#666';
                    }
                } catch (error) {
                    console.error('Error checking system health:', error);
                    indicator.classList.add('issue');
                    statusText.textContent = 'health check unavailable';
                }
            }

            async function loadPhotos(page = currentPage) {
                const requestedPage = Math.max(1, parseInt(page, 10) || 1);
                const requestId = ++latestPhotoLoadRequest;
                try {
                    await loadExtensionActions();
                    const response = await fetch(`/api/photos?page=${requestedPage}&limit=${photosPerPage}`);
                    if (!response.ok) {
                        // Server returned an error — skip this poll, will retry
                        console.log('Photos API returned', response.status, '— will retry');
                        return;
                    }
                    const data = await response.json();

                    // Ignore older refreshes that finished after a newer page request.
                    if (requestId !== latestPhotoLoadRequest) {
                        return;
                    }

                    const nextPagination = data.pagination || {};
                    const totalPages = Math.max(1, nextPagination.total_pages || 1);
                    if (requestedPage > totalPages) {
                        loadPhotos(totalPages);
                        return;
                    }

                    photos = data.photos || [];
                    pagination = nextPagination;
                    currentPage = requestedPage;
                    updatePageUrl();

                    renderGallery();
                    updatePhotoCount();
                    renderPagination();
                } catch (error) {
                    console.error('Error loading photos:', error);
                }
            }

            async function loadExtensionActions() {
                try {
                    const response = await fetch('/api/extensions/actions');
                    if (!response.ok) {
                        extensionActions = [];
                        return;
                    }
                    const data = await response.json();
                    extensionActions = data.actions || [];
                } catch (error) {
                    console.error('Error loading extension actions:', error);
                    extensionActions = [];
                }
            }
            
            async function clearScreen() {
                try {
                    const btn = document.querySelector('button[onclick="clearScreen()"]');
                    if (btn) btn.disabled = true;
                    const resp = await fetch('/api/display/clear', { method: 'POST' });
                    if (!resp.ok) throw new Error(await resp.text());
                    const data = await resp.json();
                    alert(data.message || 'screen cleared');
                } catch (error) {
                    console.error('Error clearing screen:', error);
                    alert('failed to clear screen');
                } finally {
                    const btn = document.querySelector('button[onclick="clearScreen()"]');
                    if (btn) btn.disabled = false;
                }
            }

            function renderGallery() {
                const grid = document.getElementById('photo-grid');
                
                if (photos.length === 0) {
                    grid.innerHTML = '<div class="loading">no photos found. capture your first photo!</div>';
                    return;
                }
                
                grid.innerHTML = photos.map(photo => `
                    <div class="photo-card" data-photo-id="${photo.id}">
                        <img 
                            src="${photo.dithered_path || photo.original_path}" 
                            alt="Photo ${photo.id}"
                            class="photo-image"
                        />
                        <div class="photo-info">
                            <div class="photo-actions">
                                <a href="${photo.original_path}" class="action-btn btn-primary" download onclick="event.stopPropagation()">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 48 48" style="margin-right: 5px;">
                                        <path fill="currentColor" d="M26 6a2 2 0 1 0-4 0h4Zm-3.414 37.414a2 2 0 0 0 2.828 0l12.728-12.728a2 2 0 1 0-2.828-2.828L24 39.172 12.686 27.858a2 2 0 1 0-2.828 2.828l12.728 12.728ZM24 6h-2v36h4V6h-2Z"/>
                                    </svg>
                                    original
                                </a>
                                ${photo.has_dithered ? `
                                    <a href="/api/download/dithered/${encodeURIComponent(photo.dithered_path.split('/').pop())}" class="action-btn btn-secondary" download onclick="event.stopPropagation()">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 48 48" style="margin-right: 5px;">
                                            <path fill="currentColor" d="M10 28h4v4h-4v-4Zm4 4h4v4h-4v-4Z"/>
                                            <path fill="currentColor" d="M14 32h4v4h-4v-4Zm4 4h4v4h-4v-4Zm20-8h-4v4h4v-4Zm-4 4h-4v4h4v-4Zm-4 4h-4v4h4v-4Zm-8 4h4v4h-4v-4Zm0-4h4v4h-4v-4Zm0-4h4v4h-4v-4Zm0-4h4v4h-4v-4Zm0-4h4v4h-4v-4Zm0-4h4v4h-4v-4Zm0-4h4v4h-4v-4Zm0-4h4v4h-4v-4Zm0-4h4v4h-4V8Zm0-4h4v4h-4V4Z"/>
                                        </svg>
                                        dithered
                                    </a>
                                ` : ''}
                                <button class="action-btn btn-success" onclick="event.stopPropagation(); displayPhoto('${photo.id}')">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 48 48" style="margin-right: 5px;">
                                        <path fill="currentColor" d="M6 30h12v12H6V30Zm12-12h12v12H18V18ZM30 6h12v12H30V6Zm0 24h12v12H30V30ZM6 6h12v12H6V6Z"/>
                                    </svg>
                                    display
                                </button>
                                ${renderExtensionButtons(photo)}
                            </div>
                        </div>
                    </div>
                `).join('');
                
                // Add click handlers for photo cards
                setupPhotoCardHandlers();
            }

            function renderExtensionButtons(photo) {
                return extensionActions
                    .filter(action => !action.requires_dithered || photo.has_dithered)
                    .map(action => `
                        <button class="action-btn btn-secondary" data-extension-id="${action.id}" data-photo-id="${photo.id}" onclick="event.stopPropagation(); runExtensionAction('${action.id}', '${photo.id}', this)">
                            ${action.id === 'arena' ? `
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 48 48" style="margin-right: 5px; transform: rotate(180deg);">
                                    <path fill="currentColor" d="M26 6a2 2 0 1 0-4 0h4Zm-3.414 37.414a2 2 0 0 0 2.828 0l12.728-12.728a2 2 0 1 0-2.828-2.828L24 39.172 12.686 27.858a2 2 0 1 0-2.828 2.828l12.728 12.728ZM24 6h-2v36h4V6h-2Z"/>
                                </svg>
                            ` : ''}
                            ${action.action_label}
                        </button>
                    `).join('');
            }
            
            function updatePhotoCount() {
                if (pagination.total_photos !== undefined) {
                    document.getElementById('photo-count').textContent = `${pagination.total_photos} photos`;
                } else {
                    document.getElementById('photo-count').textContent = `${photos.length} photos`;
                }
            }
            
            function renderPagination() {
                const paginationDiv = document.getElementById('pagination');
                const pageNumbersDiv = document.getElementById('page-numbers');
                const paginationInfo = document.getElementById('pagination-info');
                const prevBtn = document.getElementById('prev-btn');
                const nextBtn = document.getElementById('next-btn');
                const pageJumpInput = document.getElementById('page-jump-input');
                
                if (!pagination || pagination.total_pages <= 1) {
                    paginationDiv.style.display = 'none';
                    return;
                }
                
                paginationDiv.style.display = 'flex';
                
                // Update navigation buttons
                prevBtn.disabled = !pagination.has_prev;
                nextBtn.disabled = !pagination.has_next;
                pageJumpInput.max = pagination.total_pages;
                pageJumpInput.value = currentPage;
                
                // Update pagination info
                const startItem = ((currentPage - 1) * photosPerPage) + 1;
                const endItem = Math.min(currentPage * photosPerPage, pagination.total_photos);
                paginationInfo.textContent = `${startItem}-${endItem} of ${pagination.total_photos}`;
                
                // Generate page numbers
                pageNumbersDiv.innerHTML = '';
                const maxVisiblePages = 5;
                let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
                let endPage = Math.min(pagination.total_pages, startPage + maxVisiblePages - 1);
                
                // Adjust start page if we're near the end
                if (endPage - startPage + 1 < maxVisiblePages) {
                    startPage = Math.max(1, endPage - maxVisiblePages + 1);
                }
                
                // Add first page and ellipsis if needed
                if (startPage > 1) {
                    addPageButton(1);
                    if (startPage > 2) {
                        const ellipsis = document.createElement('span');
                        ellipsis.textContent = '...';
                        ellipsis.className = 'pagination-ellipsis';
                        pageNumbersDiv.appendChild(ellipsis);
                    }
                }
                
                // Add visible page numbers
                for (let i = startPage; i <= endPage; i++) {
                    addPageButton(i);
                }
                
                // Add last page and ellipsis if needed
                if (endPage < pagination.total_pages) {
                    if (endPage < pagination.total_pages - 1) {
                        const ellipsis = document.createElement('span');
                        ellipsis.textContent = '...';
                        ellipsis.className = 'pagination-ellipsis';
                        pageNumbersDiv.appendChild(ellipsis);
                    }
                    addPageButton(pagination.total_pages);
                }
            }
            
            function addPageButton(pageNum) {
                const button = document.createElement('button');
                button.textContent = pageNum;
                button.className = 'pagination-btn' + (pageNum === currentPage ? ' active' : '');
                button.onclick = () => changePage(pageNum);
                document.getElementById('page-numbers').appendChild(button);
            }
            
            function changePage(page) {
                if (page >= 1 && page <= pagination.total_pages && page !== currentPage) {
                    notifyUserActivity(); // Track pagination interaction
                    loadPhotos(page);
                }
            }

            function jumpToPage(event) {
                event.preventDefault();
                const input = document.getElementById('page-jump-input');
                const requestedPage = parseInt(input.value, 10);
                if (!Number.isInteger(requestedPage)) {
                    input.value = currentPage;
                    return;
                }
                const targetPage = Math.min(Math.max(requestedPage, 1), pagination.total_pages);
                if (targetPage === currentPage) {
                    input.value = currentPage;
                    return;
                }
                changePage(targetPage);
            }

            function updatePageUrl() {
                const url = new URL(window.location.href);
                if (currentPage === 1) {
                    url.searchParams.delete('page');
                } else {
                    url.searchParams.set('page', currentPage);
                }
                window.history.replaceState({}, '', url);
            }

            function setupPhotoCardHandlers() {
                const photoCards = document.querySelectorAll('.photo-card');
                
                photoCards.forEach(card => {
                    let tapTimeout;
                    let lastTap = 0;
                    
                    // Handle click/tap events
                    card.addEventListener('click', function(e) {
                        const currentTime = new Date().getTime();
                        const tapLength = currentTime - lastTap;
                        
                        // Check if this is a touch device
                        const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
                        
                        if (isTouchDevice) {
                            // On mobile: first tap shows buttons, second tap (double-tap) opens photo
                            if (tapLength < 500 && tapLength > 0) {
                                // Double tap - open photo
                                const photoId = card.getAttribute('data-photo-id');
                                viewPhoto(photoId);
                                card.classList.remove('active');
                            } else {
                                // Single tap - toggle buttons
                                clearTimeout(tapTimeout);
                                tapTimeout = setTimeout(() => {
                                    // Remove active class from all other cards
                                    photoCards.forEach(otherCard => {
                                        if (otherCard !== card) {
                                            otherCard.classList.remove('active');
                                        }
                                    });
                                    // Toggle this card
                                    card.classList.toggle('active');
                                }, 300);
                            }
                            lastTap = currentTime;
                        } else {
                            // On desktop: single click opens photo (hover shows buttons)
                            const photoId = card.getAttribute('data-photo-id');
                            viewPhoto(photoId);
                        }
                    });
                    
                    // Close buttons when clicking outside on mobile
                    document.addEventListener('click', function(e) {
                        if (!card.contains(e.target)) {
                            card.classList.remove('active');
                        }
                    });
                });
            }
            
            function formatFileSize(bytes) {
                const units = ['B', 'KB', 'MB', 'GB'];
                let size = bytes;
                let unitIndex = 0;
                
                while (size >= 1024 && unitIndex < units.length - 1) {
                    size /= 1024;
                    unitIndex++;
                }
                
                return `${size.toFixed(1)} ${units[unitIndex]}`;
            }
            
            function viewPhoto(photoId) {
                notifyUserActivity(); // Track photo viewing
                const photo = photos.find(p => p.id === photoId);
                if (photo) {
                    // Show dithered version if available, otherwise show original
                    const imageToShow = photo.has_dithered ? photo.dithered_path : photo.original_path;
                    window.open(imageToShow, '_blank');
                }
            }
            
            async function displayPhoto(photoId) {
                try {
                    notifyUserActivity(); // Track display interaction
                    const response = await fetch(`/api/display/${photoId}`, {
                        method: 'POST'
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        alert(result.message);
                    } else {
                        const error = await response.json();
                        alert(`Error: ${error.detail}`);
                    }
                } catch (error) {
                    console.error('Error displaying photo:', error);
                    alert('Error displaying photo');
                }
            }

            async function showDashboardQr() {
                try {
                    notifyUserActivity();
                    const response = await fetch('/api/dashboard/qr', {
                        method: 'POST'
                    });
                    const result = await response.json();
                    if (response.ok && result.success) {
                        alert(result.message || 'Dashboard QR displayed');
                    } else {
                        alert(`Error: ${result.detail || result.message || 'Could not show dashboard QR'}`);
                    }
                } catch (error) {
                    console.error('Error showing dashboard QR:', error);
                    alert('Error showing dashboard QR');
                }
            }

            async function runExtensionAction(extensionId, photoId, button) {
                const originalText = button ? button.textContent : '';
                try {
                    notifyUserActivity();
                    if (button) {
                        button.textContent = 'uploading...';
                        button.disabled = true;
                    }

                    const response = await fetch(`/api/extensions/${extensionId}/photos/${photoId}`, {
                        method: 'POST'
                    });
                    const result = await response.json();

                    if (response.ok) {
                        alert(result.message || 'Upload complete');
                    } else {
                        alert(`Error: ${result.detail || 'Upload failed'}`);
                    }
                } catch (error) {
                    console.error('Error running extension action:', error);
                    alert('Error running extension action');
                } finally {
                    if (button) {
                        button.textContent = originalText;
                        button.disabled = false;
                    }
                }
            }
            
            async function capturePhoto() {
                try {
                    notifyUserActivity(); // Track capture interaction
                    // Find the capture button and show loading indicator
                    const captureBtn = document.querySelector('button[onclick="capturePhoto()"]');
                    if (captureBtn) {
                        captureBtn.textContent = 'capturing...';
                        captureBtn.disabled = true;
                    }
                    
                    const response = await fetch('/api/capture', {
                        method: 'POST'
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        alert(result.message);
                        loadPhotos(currentPage);
                    } else {
                        const error = await response.json();
                        alert(`Error: ${error.detail}`);
                    }
                } catch (error) {
                    console.error('Error capturing photo:', error);
                    alert('Error capturing photo');
                } finally {
                    updateHealth();
                    // Restore button state
                    const captureBtn = document.querySelector('button[onclick="capturePhoto()"]');
                    if (captureBtn) {
                        captureBtn.textContent = 'capture photo';
                        captureBtn.disabled = false;
                    }
                }
            }
            
            async function reprocessPhoto(photoId) {
                try {
                    const response = await fetch(`/api/reprocess/${photoId}`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        }
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        alert(result.message);
                        // Refresh gallery to show updated photo (stay on current page)
                        loadPhotos(currentPage);
                    } else {
                        const error = await response.json();
                        alert(`Error: ${error.detail}`);
                    }
                } catch (error) {
                    console.error('Error reprocessing photo:', error);
                    alert('Error reprocessing photo');
                }
            }
            
            async function openSettings() {
                try {
                    notifyUserActivity(); // Track settings interaction
                    const response = await fetch('/api/settings');
                    const settings = await response.json();
                    populateSettingsForm(settings);
                    updateHealth();
                    document.getElementById('settings-modal').style.display = 'block';
                    lockSettingsPageScroll();
                } catch (error) {
                    console.error('Error loading settings:', error);
                    alert('Error loading settings');
                }
            }
            
            function closeSettings(force = false) {
                if (!force && hasUnsavedSettings()) {
                    const shouldClose = confirm('You have unsaved settings. Close without saving?');
                    if (!shouldClose) {
                        return false;
                    }
                }
                document.getElementById('settings-modal').style.display = 'none';
                unlockSettingsPageScroll();
                settingsFormSnapshot = null;
                return true;
            }

            function lockSettingsPageScroll() {
                if (document.body.classList.contains('settings-open')) {
                    return;
                }
                settingsPageScrollY = window.scrollY;
                document.documentElement.classList.add('settings-open');
                document.body.classList.add('settings-open');
                document.body.style.top = `-${settingsPageScrollY}px`;
            }

            function unlockSettingsPageScroll() {
                if (!document.body.classList.contains('settings-open')) {
                    return;
                }
                document.documentElement.classList.remove('settings-open');
                document.body.classList.remove('settings-open');
                document.body.style.top = '';
                window.scrollTo(0, settingsPageScrollY);
            }

            function getSettingsFormSnapshot() {
                const controls = document.querySelectorAll(
                    '#settings-modal input.setting-input, #settings-modal select.setting-input'
                );
                return JSON.stringify({
                    values: Array.from(controls).map(control => [control.id, control.value]),
                    arenaTokenShouldClear
                });
            }

            function hasUnsavedSettings() {
                return settingsFormSnapshot !== null
                    && settingsFormSnapshot !== getSettingsFormSnapshot();
            }
            
            function populateSettingsForm(settings) {
                // Camera settings
                document.getElementById('resolution-width').value = settings.camera.resolution.width;
                document.getElementById('resolution-height').value = settings.camera.resolution.height;
                document.getElementById('exposure-value').value = settings.camera.exposure_value;
                document.getElementById('sharpness').value = settings.camera.sharpness;
                document.getElementById('autofocus-mode').value = settings.camera.autofocus_mode;
                
                // Processing settings
                document.getElementById('saturation').value = settings.processing.saturation;
                document.getElementById('brightness-factor').value = settings.processing.brightness_factor;
                document.getElementById('color-factor').value = settings.processing.color_factor;
                document.getElementById('dithering-method').value = settings.processing.dithering_method;
                document.getElementById('bayer-size').value = settings.processing.bayer_size || 4;
                document.getElementById('threshold-scale').value = settings.processing.threshold_scale || 1.0;
                
                // Show/hide ordered dithering settings
                toggleOrderedSettings();
                
                // System settings
                document.getElementById('fast-refresh').value = settings.display?.fast_refresh === true ? 'true' : 'false';
                document.getElementById('camera-name').value = settings.system.camera_name || '';
                document.getElementById('auto-refresh-interval').value = settings.system.auto_refresh_interval;
                document.getElementById('auto-timeout-enabled').value = settings.system.auto_timeout_enabled ? 'true' : 'false';
                document.getElementById('auto-timeout-minutes').value = settings.system.auto_timeout_minutes || 10;
                document.getElementById('show-dashboard-qr-on-wifi-connect').value = settings.system.show_dashboard_qr_on_wifi_connect !== false ? 'true' : 'false';
                const exportSettings = settings.exports || {};
                document.getElementById('upscale-dithered-2x').value = exportSettings.upscale_dithered_2x ? 'true' : 'false';
                document.getElementById('update-status').textContent = 'Updates code, dependencies, and service files while preserving settings and photos.';
                document.getElementById('update-install-btn').style.display = 'none';

                const arenaSettings = (settings.extensions && settings.extensions.arena) || {};
                document.getElementById('arena-enabled').value = arenaSettings.enabled ? 'true' : 'false';
                document.getElementById('arena-channel').value = arenaSettings.channel || '';
                document.getElementById('arena-access-token').value = '';
                arenaTokenShouldClear = false;
                updateArenaTokenStatus(Boolean(arenaSettings.access_token_configured));
                settingsFormSnapshot = getSettingsFormSnapshot();
            }

            function updateArenaTokenStatus(configured) {
                const status = document.getElementById('arena-token-status');
                const clearBtn = document.getElementById('arena-clear-token-btn');
                if (arenaTokenShouldClear) {
                    status.textContent = 'Token will be cleared when settings are saved';
                    clearBtn.disabled = true;
                } else if (configured) {
                    status.textContent = 'Token saved. Leave blank to keep it.';
                    clearBtn.disabled = false;
                } else {
                    status.textContent = 'No token configured';
                    clearBtn.disabled = true;
                }
            }

            function clearArenaToken() {
                arenaTokenShouldClear = true;
                document.getElementById('arena-access-token').value = '';
                updateArenaTokenStatus(false);
            }

            function updateSoftwareStatus(data) {
                const status = document.getElementById('update-status');
                const installBtn = document.getElementById('update-install-btn');
                status.textContent = data.message || 'Update status checked.';
                installBtn.style.display = data.can_update ? 'inline-block' : 'none';
            }

            async function checkForUpdates() {
                const status = document.getElementById('update-status');
                const checkBtn = document.getElementById('update-check-btn');
                const installBtn = document.getElementById('update-install-btn');

                try {
                    checkBtn.disabled = true;
                    installBtn.style.display = 'none';
                    status.textContent = 'Checking for updates...';
                    const response = await fetch('/api/update/status');
                    const data = await response.json().catch(() => ({}));

                    if (response.ok) {
                        updateSoftwareStatus(data);
                    } else {
                        status.textContent = data.detail || 'Could not check for updates.';
                    }
                } catch (error) {
                    console.error('Error checking for updates:', error);
                    status.textContent = 'Could not check for updates.';
                } finally {
                    checkBtn.disabled = false;
                }
            }

            async function installUpdate() {
                const status = document.getElementById('update-status');
                const checkBtn = document.getElementById('update-check-btn');
                const installBtn = document.getElementById('update-install-btn');

                if (!confirm('Install the available update? The camera should be rebooted after the update finishes.')) {
                    return;
                }

                try {
                    checkBtn.disabled = true;
                    installBtn.disabled = true;
                    status.textContent = 'Installing update...';
                    const response = await fetch('/api/update/install', { method: 'POST' });
                    const data = await response.json().catch(() => ({}));

                    if (response.ok) {
                        status.textContent = data.message || 'Update installed. Reboot the camera to finish.';
                        installBtn.style.display = 'none';
                    } else {
                        status.textContent = data.detail || 'Could not install update.';
                        if (data.can_update) {
                            installBtn.style.display = 'inline-block';
                        }
                    }
                } catch (error) {
                    console.error('Error installing update:', error);
                    status.textContent = 'Could not install update.';
                } finally {
                    checkBtn.disabled = false;
                    installBtn.disabled = false;
                }
            }
            
            async function saveSettings() {
                try {
                    const settings = {
                        camera: {
                            resolution: {
                                width: parseInt(document.getElementById('resolution-width').value),
                                height: parseInt(document.getElementById('resolution-height').value)
                            },
                            exposure_value: parseFloat(document.getElementById('exposure-value').value),
                            sharpness: parseInt(document.getElementById('sharpness').value),
                            autofocus_mode: parseInt(document.getElementById('autofocus-mode').value)
                        },
                        processing: {
                            saturation: parseFloat(document.getElementById('saturation').value),
                            brightness_factor: parseFloat(document.getElementById('brightness-factor').value),
                            color_factor: parseFloat(document.getElementById('color-factor').value),
                            dithering_method: document.getElementById('dithering-method').value,
                            bayer_size: parseInt(document.getElementById('bayer-size').value),
                            threshold_scale: parseFloat(document.getElementById('threshold-scale').value)
                        },
                        display: {
                            fast_refresh: document.getElementById('fast-refresh').value === 'true'
                        },
                        system: {
                            camera_name: document.getElementById('camera-name').value.trim(),
                            auto_refresh_interval: parseInt(document.getElementById('auto-refresh-interval').value),
                            auto_timeout_enabled: document.getElementById('auto-timeout-enabled').value === 'true',
                            auto_timeout_minutes: parseInt(document.getElementById('auto-timeout-minutes').value),
                            show_dashboard_qr_on_wifi_connect: document.getElementById('show-dashboard-qr-on-wifi-connect').value === 'true'
                        },
                        exports: {
                            upscale_dithered_2x: document.getElementById('upscale-dithered-2x').value === 'true'
                        },
                        extensions: {
                            arena: {
                                enabled: document.getElementById('arena-enabled').value === 'true',
                                channel: document.getElementById('arena-channel').value.trim(),
                                access_token: document.getElementById('arena-access-token').value.trim(),
                                access_token_clear: arenaTokenShouldClear
                            }
                        }
                    };
                    
                    const response = await fetch('/api/settings', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(settings)
                    });
                    
                    if (response.ok) {
                        settingsFormSnapshot = getSettingsFormSnapshot();
                        alert('Settings saved successfully!');
                        closeSettings(true);
                        // Update auto-refresh interval if changed
                        updateAutoRefreshInterval();
                        await loadExtensionActions();
                        renderGallery();
                    } else {
                        alert('Error saving settings');
                    }
                } catch (error) {
                    console.error('Error saving settings:', error);
                    alert('Error saving settings');
                }
            }
            
            async function resetSettings() {
                if (confirm('Are you sure you want to reset all settings to defaults?')) {
                    try {
                        const defaultSettings = {
                            camera: {
                                resolution: {width: 1200, height: 800},
                                exposure_value: 0,
                                sharpness: 3,
                                autofocus_mode: 2
                            },
                            processing: {
                                saturation: 0.6,
                                brightness_factor: 1.1,
                                color_factor: 1.4,
                                dithering_method: "floyd_steinberg",
                                bayer_size: 4,
                                threshold_scale: 1.0
                            },
                            display: {
                                fast_refresh: false
                            },
                            system: {
                                camera_name: "",
                                auto_refresh_interval: 30,
                                auto_timeout_enabled: true,
                                auto_timeout_minutes: 10,
                                show_dashboard_qr_on_wifi_connect: true
                            },
                            exports: {
                                upscale_dithered_2x: false
                            },
                            extensions: {
                                arena: {
                                    enabled: false,
                                    channel: "",
                                    access_token: "",
                                    access_token_clear: true
                                }
                            }
                        };
                        
                        const response = await fetch('/api/settings', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify(defaultSettings)
                        });
                        
                        if (response.ok) {
                            populateSettingsForm(defaultSettings);
                            await loadExtensionActions();
                            renderGallery();
                            alert('Settings reset to defaults!');
                        } else {
                            alert('Error resetting settings');
                        }
                    } catch (error) {
                        console.error('Error resetting settings:', error);
                        alert('Error resetting settings');
                    }
                }
            }
            
            let autoRefreshInterval;
            
            function updateAutoRefreshInterval() {
                // Clear existing interval
                if (autoRefreshInterval) {
                    clearInterval(autoRefreshInterval);
                }
                
                // Get current auto-refresh setting
                fetch('/api/settings')
                    .then(response => response.json())
                    .then(settings => {
                        const intervalSeconds = settings.system.auto_refresh_interval;
                        if (intervalSeconds > 0) {
                            autoRefreshInterval = setInterval(
                                () => loadPhotos(currentPage),
                                intervalSeconds * 1000
                            );
                        }
                    })
                    .catch(error => console.error('Error updating auto-refresh:', error));
            }
            
            function toggleOrderedSettings() {
                const ditheringMethod = document.getElementById('dithering-method').value;
                const bayerSettings = document.getElementById('bayer-settings');
                const thresholdSettings = document.getElementById('threshold-settings');
                
                if (ditheringMethod === 'ordered') {
                    bayerSettings.style.display = 'block';
                    thresholdSettings.style.display = 'block';
                } else {
                    bayerSettings.style.display = 'none';
                    thresholdSettings.style.display = 'none';
                }
            }
            
            function refreshGallery() {
                notifyUserActivity(); // Track refresh interaction
                loadPhotos(currentPage);
            }

            function formatDownloadSize(bytes) {
                if (!Number.isFinite(bytes) || bytes <= 0) {
                    return '';
                }
                const units = ['B', 'KB', 'MB', 'GB'];
                const unitIndex = Math.min(
                    Math.floor(Math.log(bytes) / Math.log(1024)),
                    units.length - 1
                );
                const value = bytes / Math.pow(1024, unitIndex);
                return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
            }

            function resetDownloadButton(downloadBtn) {
                downloadBtn.dataset.ready = 'false';
                delete downloadBtn.dataset.downloadSize;
                downloadBtn.textContent = 'download all photos';
                downloadBtn.disabled = false;
                downloadBtn.style.opacity = '1';
            }

            function handleDownloadError(downloadBtn, abortBtn, error) {
                console.error('Error downloading photos:', error);
                alert(`Error: ${error.message}`);
                if (downloadBtn) {
                    resetDownloadButton(downloadBtn);
                }
                if (abortBtn) {
                    abortBtn.style.display = 'none';
                }
            }

            function startPreparedZipDownload(downloadBtn, size = '') {
                downloadBtn.dataset.ready = 'false';
                downloadBtn.textContent = size
                    ? `downloading ${size} in browser...`
                    : 'downloading in browser...';
                downloadBtn.disabled = true;
                downloadBtn.style.opacity = '0.7';

                const link = document.createElement('a');
                link.href = '/api/photos/download-all/result';
                link.download = `reframe-photos-${new Date().toISOString().split('T')[0]}.zip`;
                link.style.display = 'none';
                document.body.appendChild(link);
                link.click();
                link.remove();

                monitorBrowserDownload(downloadBtn);
            }

            async function monitorBrowserDownload(downloadBtn, attempts = 0) {
                try {
                    const response = await fetch('/api/photos/download-all/progress');
                    if (!response.ok) {
                        throw new Error('Could not check download status');
                    }
                    const progress = await response.json();

                    if (progress.status === 'idle') {
                        downloadBtn.textContent = 'download sent to browser';
                        setTimeout(() => resetDownloadButton(downloadBtn), 2000);
                        return;
                    }
                    if (progress.status === 'completed' && attempts >= 3) {
                        const size = formatDownloadSize(progress.size_bytes);
                        downloadBtn.textContent = size
                            ? `download ZIP (${size})`
                            : 'download ZIP';
                        downloadBtn.dataset.ready = 'true';
                        downloadBtn.dataset.downloadSize = size;
                        downloadBtn.disabled = false;
                        downloadBtn.style.opacity = '1';
                        return;
                    }
                    if (progress.status === 'error' || progress.status === 'aborted') {
                        throw new Error(progress.message || 'Download failed');
                    }
                    if (attempts >= 600) {
                        throw new Error('Browser download timed out');
                    }

                    setTimeout(() => monitorBrowserDownload(downloadBtn, attempts + 1), 1000);
                } catch (error) {
                    console.error('Browser download error:', error);
                    downloadBtn.textContent = 'download failed';
                    setTimeout(() => resetDownloadButton(downloadBtn), 2500);
                }
            }
            
            async function downloadAllPhotos() {
                try {
                    notifyUserActivity(); // Track download interaction
                    
                    // Find the download button and show loading state
                    const downloadBtn = document.querySelector('button[onclick="downloadAllPhotos()"]');
                    const abortBtn = document.getElementById('abort-btn');
                    if (downloadBtn && downloadBtn.dataset.ready === 'true') {
                        startPreparedZipDownload(downloadBtn, downloadBtn.dataset.downloadSize || '');
                        return;
                    }
                    if (downloadBtn) {
                        downloadBtn.textContent = 'starting download...';
                        downloadBtn.disabled = true;
                        downloadBtn.style.opacity = '0.7';
                    }
                    
                    // Show abort button
                    if (abortBtn) {
                        abortBtn.style.display = 'inline-block';
                    }
                    
                    // Start the download process
                    const startResponse = await fetch('/api/photos/download-all/start', {
                        method: 'POST'
                    });
                    
                    if (!startResponse.ok) {
                        const error = await startResponse.json();
                        throw new Error(error.detail || 'Failed to start download');
                    }
                    
                    const startData = await startResponse.json();
                    const totalPhotos = startData.total_photos;
                    
                    // Poll for progress
                    let attempts = 0;
                    const maxAttempts = 600; // 10 minutes max
                    
                    window.progressInterval = setInterval(async () => {
                        attempts++;
                        
                        try {
                            const progressResponse = await fetch('/api/photos/download-all/progress');
                            if (progressResponse.ok) {
                                const progress = await progressResponse.json();
                                
                                if (downloadBtn) {
                                    if (progress.status === 'creating') {
                                        const percent = Math.round((progress.processed / progress.total) * 100);
                                        downloadBtn.textContent = `${progress.message} (${percent}%)`;
                                    } else if (progress.status === 'completed') {
                                        clearInterval(window.progressInterval);
                                        window.progressInterval = null;
                                        
                                        // Hide abort button
                                        if (abortBtn) {
                                            abortBtn.style.display = 'none';
                                        }

                                        const size = formatDownloadSize(progress.size_bytes);
                                        startPreparedZipDownload(downloadBtn, size);
                                    } else if (progress.status === 'aborted') {
                                        clearInterval(window.progressInterval);
                                        window.progressInterval = null;
                                        downloadBtn.textContent = 'download aborted';
                                        setTimeout(() => {
                                            if (downloadBtn) {
                                                resetDownloadButton(downloadBtn);
                                            }
                                            if (abortBtn) {
                                                abortBtn.style.display = 'none';
                                            }
                                        }, 2000);
                                        return;
                                    } else if (progress.status === 'error') {
                                        throw new Error(progress.message);
                                    }
                                }
                            } else {
                                throw new Error('Failed to get progress');
                            }
                        } catch (error) {
                            clearInterval(window.progressInterval);
                            window.progressInterval = null;
                            handleDownloadError(downloadBtn, abortBtn, error);
                            return;
                        }
                        
                        // Timeout after max attempts
                        if (attempts >= maxAttempts) {
                            clearInterval(window.progressInterval);
                            window.progressInterval = null;
                            handleDownloadError(
                                downloadBtn,
                                abortBtn,
                                new Error('Download timed out after 10 minutes')
                            );
                        }
                    }, 1000); // Check progress every second
                    
                } catch (error) {
                    const downloadBtn = document.querySelector('button[onclick="downloadAllPhotos()"]');
                    const abortBtn = document.getElementById('abort-btn');
                    handleDownloadError(downloadBtn, abortBtn, error);
                }
            }
            
            async function abortDownload() {
                try {
                    const response = await fetch('/api/photos/download-all/abort', {
                        method: 'POST'
                    });
                    
                    if (response.ok) {
                        const abortBtn = document.getElementById('abort-btn');
                        const downloadBtn = document.querySelector('button[onclick="downloadAllPhotos()"]');
                        
                        if (abortBtn) {
                            abortBtn.textContent = 'aborting...';
                            abortBtn.disabled = true;
                        }
                        
                        // Clear any existing progress interval
                        if (window.progressInterval) {
                            clearInterval(window.progressInterval);
                            window.progressInterval = null;
                        }
                        
                        // Reset download button after a short delay
                        setTimeout(() => {
                            if (downloadBtn) {
                                downloadBtn.textContent = 'download all photos';
                                downloadBtn.disabled = false;
                                downloadBtn.style.opacity = '1';
                            }
                            if (abortBtn) {
                                abortBtn.style.display = 'none';
                            }
                        }, 1000);
                        
                    } else {
                        alert('Failed to abort download');
                    }
                } catch (error) {
                    console.error('Error aborting download:', error);
                    alert('Error aborting download');
                }
            }
            
            async function deleteAllPhotos() {
                const confirmed = confirm('⚠️ This will permanently delete ALL photos from the system. This action cannot be undone. Are you absolutely sure?');
                if (!confirmed) {
                    return;
                }
                
                const doubleConfirmed = confirm('Final confirmation: Delete ALL photos? This will remove both original and dithered versions.');
                if (!doubleConfirmed) {
                    return;
                }
                
                try {
                    notifyUserActivity(); // Track delete interaction
                    
                    // Find the delete button and show loading state
                    const deleteBtn = document.querySelector('button[onclick="deleteAllPhotos()"]');
                    const abortDeleteBtn = document.getElementById('abort-delete-btn');
                    if (deleteBtn) {
                        const originalText = deleteBtn.textContent;
                        deleteBtn.textContent = 'starting deletion...';
                        deleteBtn.disabled = true;
                        deleteBtn.style.opacity = '0.7';
                    }
                    
                    // Show abort button
                    if (abortDeleteBtn) {
                        abortDeleteBtn.style.display = 'inline-block';
                    }
                    
                    // Start the delete process
                    const startResponse = await fetch('/api/photos/delete-all/start', {
                        method: 'POST'
                    });
                    
                    if (!startResponse.ok) {
                        const error = await startResponse.json();
                        throw new Error(error.detail || 'Failed to start deletion');
                    }
                    
                    const startData = await startResponse.json();
                    
                    // If no photos to delete, show message and return
                    if (startData.status === 'completed') {
                        alert(startData.message || 'No photos to delete');
                        if (deleteBtn) {
                            deleteBtn.textContent = originalText;
                            deleteBtn.disabled = false;
                            deleteBtn.style.opacity = '1';
                        }
                        if (abortDeleteBtn) {
                            abortDeleteBtn.style.display = 'none';
                        }
                        return;
                    }
                    
                    const totalPhotos = startData.total_photos;
                    
                    // Poll for progress
                    let attempts = 0;
                    const maxAttempts = 300; // 5 minutes max
                    
                    window.deleteProgressInterval = setInterval(async () => {
                        attempts++;
                        
                        try {
                            const progressResponse = await fetch('/api/photos/delete-all/progress');
                            if (progressResponse.ok) {
                                const progress = await progressResponse.json();
                                
                                if (deleteBtn) {
                                    if (progress.status === 'deleting') {
                                        const percent = Math.round((progress.processed / progress.total) * 100);
                                        deleteBtn.textContent = `${progress.message} (${percent}%)`;
                                    } else if (progress.status === 'completed') {
                                        deleteBtn.textContent = 'deletion complete!';
                                        clearInterval(window.deleteProgressInterval);
                                        window.deleteProgressInterval = null;
                                        
                                        // Hide abort button
                                        if (abortDeleteBtn) {
                                            abortDeleteBtn.style.display = 'none';
                                        }
                                        
                                        // Show success message and refresh gallery
                                        alert(progress.message || 'All photos deleted successfully');
                                        loadPhotos(1);
                                        
                                        setTimeout(() => {
                                            if (deleteBtn) {
                                                deleteBtn.textContent = originalText;
                                                deleteBtn.disabled = false;
                                                deleteBtn.style.opacity = '1';
                                            }
                                        }, 2000);
                                    } else if (progress.status === 'aborted') {
                                        clearInterval(window.deleteProgressInterval);
                                        window.deleteProgressInterval = null;
                                        deleteBtn.textContent = 'deletion aborted';
                                        setTimeout(() => {
                                            if (deleteBtn) {
                                                deleteBtn.textContent = originalText;
                                                deleteBtn.disabled = false;
                                                deleteBtn.style.opacity = '1';
                                            }
                                            if (abortDeleteBtn) {
                                                abortDeleteBtn.style.display = 'none';
                                            }
                                        }, 2000);
                                        return;
                                    } else if (progress.status === 'error') {
                                        throw new Error(progress.message);
                                    }
                                }
                            } else {
                                throw new Error('Failed to get progress');
                            }
                        } catch (error) {
                            clearInterval(window.deleteProgressInterval);
                            window.deleteProgressInterval = null;
                            throw error;
                        }
                        
                        // Timeout after max attempts
                        if (attempts >= maxAttempts) {
                            clearInterval(window.deleteProgressInterval);
                            window.deleteProgressInterval = null;
                            throw new Error('Deletion timed out after 5 minutes');
                        }
                    }, 1000); // Check progress every second
                    
                } catch (error) {
                    console.error('Error deleting photos:', error);
                    alert(`Error: ${error.message}`);
                    
                    // Reset button on error
                    const deleteBtn = document.querySelector('button[onclick="deleteAllPhotos()"]');
                    const abortDeleteBtn = document.getElementById('abort-delete-btn');
                    if (deleteBtn) {
                        deleteBtn.textContent = 'delete all photos';
                        deleteBtn.disabled = false;
                        deleteBtn.style.opacity = '1';
                    }
                    if (abortDeleteBtn) {
                        abortDeleteBtn.style.display = 'none';
                    }
                }
            }
            
            async function abortDelete() {
                try {
                    const response = await fetch('/api/photos/delete-all/abort', {
                        method: 'POST'
                    });
                    
                    if (response.ok) {
                        const abortDeleteBtn = document.getElementById('abort-delete-btn');
                        const deleteBtn = document.querySelector('button[onclick="deleteAllPhotos()"]');
                        
                        if (abortDeleteBtn) {
                            abortDeleteBtn.textContent = 'aborting...';
                            abortDeleteBtn.disabled = true;
                        }
                        
                        // Clear any existing progress interval
                        if (window.deleteProgressInterval) {
                            clearInterval(window.deleteProgressInterval);
                            window.deleteProgressInterval = null;
                        }
                        
                        // Reset delete button after a short delay
                        setTimeout(() => {
                            if (deleteBtn) {
                                deleteBtn.textContent = 'delete all photos';
                                deleteBtn.disabled = false;
                                deleteBtn.style.opacity = '1';
                            }
                            if (abortDeleteBtn) {
                                abortDeleteBtn.style.display = 'none';
                            }
                        }, 1000);
                        
                    } else {
                        alert('Failed to abort deletion');
                    }
                } catch (error) {
                    console.error('Error aborting deletion:', error);
                    alert('Error aborting deletion');
                }
            }
            
            // Load photos on page load
            document.addEventListener('DOMContentLoaded', function() {
                loadPhotos(currentPage);
                updateAutoRefreshInterval();
                updateHealth();
                
                // Add event listener for dithering method changes
                document.getElementById('dithering-method').addEventListener('change', toggleOrderedSettings);
                
                // Keep the home indicator and settings health panel current.
                setInterval(updateHealth, 30000);
            });

            window.addEventListener('beforeunload', function(event) {
                const modal = document.getElementById('settings-modal');
                if (modal.style.display === 'block' && hasUnsavedSettings()) {
                    event.preventDefault();
                    event.returnValue = '';
                }
            });
            
            // Close modal when clicking outside of it
            window.onclick = function(event) {
                const modal = document.getElementById('settings-modal');
                if (event.target === modal) {
                    closeSettings();
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/photos")
async def list_photos(page: int = 1, limit: int = 20):
    """Get paginated list of photos with metadata from the hardware service."""
    if page < 1:
        page = 1
    if limit < 1 or limit > 100:
        limit = 20
    try:
        # Fetch all photos from hardware service and paginate here for simplicity
        all_photos = await reframe_client.get("/photos")
        # Rewrite absolute file system paths to dashboard-served URLs
        for photo in all_photos:
            try:
                if photo.get("original_path"):
                    from os.path import basename as _bn
                    orig_name = _bn(photo["original_path"])
                    photo["original_path"] = f"/photos/{orig_name}"
                if photo.get("dithered_path"):
                    from os.path import basename as _bn
                    dith_name = _bn(photo["dithered_path"])
                    photo["dithered_path"] = f"/dithered/{dith_name}"
            except Exception:
                continue
    except Exception as e:
        logging.warning(f"Error fetching photos from hardware service: {e}")
        all_photos = []
    start = (page - 1) * limit
    end = start + limit
    total = len(all_photos)
    total_pages = (total + limit - 1) // limit if total else 1
    return {
        "photos": all_photos[start:end],
        "pagination": {
            "page": page,
            "limit": limit,
            "total_photos": total,
            "total_pages": total_pages,
            "has_prev": page > 1,
            "has_next": page < total_pages,
        },
    }

@app.get("/api/photos/{photo_id}")
async def get_photo_info(photo_id: str):
    """Get information about a specific photo from the hardware service."""
    try:
        photo = await reframe_client.get(f"/photos/{photo_id}")
        # Rewrite paths to URLs served by this dashboard
        from os.path import basename as _bn
        if photo.get("original_path"):
            photo["original_path"] = f"/photos/{_bn(photo['original_path'])}"
        if photo.get("dithered_path"):
            photo["dithered_path"] = f"/dithered/{_bn(photo['dithered_path'])}"
        return photo
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/photos/{filename}")
async def serve_original_photo(filename: str):
    """Serve original photo file."""
    file_path = os.path.join(PHOTOS_PATH, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Photo not found")
    return FileResponse(file_path)

@app.get("/dithered/{filename}")
async def serve_dithered_photo(filename: str):
    """Serve dithered photo file."""
    file_path = os.path.join(DITHERED_PHOTOS_PATH, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dithered photo not found")
    return FileResponse(file_path)


@app.get("/api/download/dithered/{filename}")
async def download_dithered_photo(filename: str):
    """Download one dithered photo with optional on-demand 2x scaling."""
    if filename != os.path.basename(filename):
        raise HTTPException(status_code=400, detail="Invalid photo filename")

    file_path = os.path.join(DITHERED_PHOTOS_PATH, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dithered photo not found")

    settings = settings_manager.load_settings()
    upscale_2x = bool(settings.get("exports", {}).get("upscale_dithered_2x", False))
    if not upscale_2x:
        return FileResponse(file_path, filename=filename)

    try:
        content, export_filename, media_type = await asyncio.to_thread(
            prepare_dithered_export,
            file_path,
            True
        )
    except OSError as error:
        raise HTTPException(status_code=500, detail=f"Could not prepare dithered download: {error}") from error

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{export_filename}"'}
    )

@app.get("/api/settings")
async def get_settings():
    """Get current settings."""
    return settings_manager.load_public_settings()

@app.post("/api/settings")
async def update_settings(request: Request):
    """Update settings and notify the hardware service to reload/apply."""
    try:
        settings_data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Settings body must be valid JSON")

    previous_settings = settings_manager.load_settings()
    try:
        success = settings_manager.save_settings(settings_data)
    except SettingsValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not success:
        raise HTTPException(status_code=500, detail="Failed to save settings")

    try:
        await reframe_client.post("/settings/reload")
    except Exception as apply_error:
        try:
            settings_manager.replace_settings(previous_settings)
            await reframe_client.post("/settings/reload")
        except Exception as rollback_error:
            logging.error(f"Settings rollback failed: {rollback_error}")
        raise HTTPException(
            status_code=502,
            detail=f"Camera rejected the settings; previous settings were restored: {apply_error}"
        )

    return {"status": "success", "message": "Settings updated successfully"}

@app.get("/api/update/status")
async def update_status():
    """Check whether the git checkout has a fast-forward update available."""
    return await get_update_status(fetch=True)

@app.post("/api/update/install")
async def install_update():
    """Pull a fast-forward update and apply its dependencies and service files."""
    status = await get_update_status(fetch=True)

    if status["up_to_date"] and not status["post_install_pending"]:
        return {
            **status,
            "status": "success",
            "message": "Software is already up to date."
        }

    if status["diverged"]:
        raise HTTPException(
            status_code=409,
            detail="This checkout has diverged from upstream and cannot be updated automatically"
        )

    if status["has_tracked_changes"]:
        raise HTTPException(
            status_code=409,
            detail="Local code changes must be committed or discarded before updating"
        )

    if not status["update_available"] and not status["post_install_pending"]:
        raise HTTPException(status_code=409, detail="No automatic update is available")

    preserved_paths = ignored_user_data_paths()
    backup_dir = None
    pull_output = ""

    if not os.path.exists(UPDATE_HELPER_PATH):
        raise HTTPException(
            status_code=501,
            detail="Update installer is missing. Run install.sh once over SSH to enable complete updates."
        )

    if status["update_available"]:
        backup_dir = backup_settings_for_update()
        upstream = status["upstream"]
        pull_args = ["pull", "--ff-only"]
        if "/" in upstream:
            remote, branch = upstream.split("/", 1)
            pull_args.extend([remote, branch])

        pull_result = await run_git(pull_args, timeout=180)
        if pull_result["returncode"] != 0:
            detail = git_error_detail(pull_result, "git pull failed")
            raise HTTPException(status_code=500, detail=f"Could not install update: {detail}")
        pull_output = pull_result["stdout"]

        pending_path = Path(UPDATE_PENDING_PATH)
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text(status["remote_revision"] + "\n", encoding="utf-8")

    apply_result = await run_repo_command(
        ["sudo", "-n", UPDATE_HELPER_PATH],
        timeout=300
    )
    if apply_result["returncode"] != 0:
        detail = git_error_detail(apply_result, "post-update installation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Code was updated, but installation did not finish: {detail}"
        )

    try:
        os.unlink(UPDATE_PENDING_PATH)
    except FileNotFoundError:
        pass

    new_status = await get_update_status(fetch=False)
    message = "Update installed. Reboot the camera to finish."
    if backup_dir:
        message = f"{message} Settings backup: {backup_dir}"

    return {
        **new_status,
        "status": "success",
        "message": message,
        "backup_dir": backup_dir,
        "preserved_paths": preserved_paths,
        "pull_output": pull_output,
        "apply_output": apply_result["stdout"]
    }

@app.get("/api/extensions/actions")
async def get_extension_actions():
    """Get extension photo actions safe for the dashboard to render."""
    settings = settings_manager.load_settings()
    return {"actions": extension_registry.public_actions(settings)}

@app.post("/api/extensions/{extension_id}/photos/{photo_id}")
async def run_extension_action(extension_id: str, photo_id: str):
    """Run an enabled extension against one photo."""
    extension = extension_registry.get(extension_id)
    if extension is None:
        raise HTTPException(status_code=404, detail="Extension not found")

    settings = settings_manager.load_settings()
    if not extension.configured(settings):
        raise HTTPException(status_code=400, detail=f"{extension.label} extension is not fully configured")

    try:
        photo = await reframe_client.get(f"/photos/{photo_id}")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Photo not found: {e}")

    result = await extension.run(photo, settings)
    return result

@app.get("/api/settings/camera")
async def get_camera_settings():
    """Get camera-specific settings."""
    return settings_manager.get_camera_settings()

@app.get("/api/settings/processing") 
async def get_processing_settings():
    """Get processing-specific settings."""
    return settings_manager.get_processing_settings()

@app.post("/api/capture")
async def capture_photo(background_tasks: BackgroundTasks):
    """Capture a new photo with current settings."""
    try:
        result = await reframe_client.post("/capture")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/display/clear")
async def clear_display():
    """Proxy to clear the e-ink display on the hardware service."""
    try:
        resp = await reframe_client.post("/display/clear")
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear display: {str(e)}")

@app.post("/api/display/{photo_id}")
async def display_photo_on_screen(photo_id: str):
    """Display a specific photo on the e-ink screen."""
    try:
        result = await reframe_client.post(f"/display/{photo_id}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dashboard/qr")
async def show_dashboard_qr():
    """Display the dashboard access QR on the e-ink screen."""
    try:
        result = await reframe_client.post("/dashboard/qr")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to show dashboard QR: {str(e)}")

@app.post("/api/reprocess/{photo_id}")
async def reprocess_photo(photo_id: str, request: Request):
    """Reprocess an existing photo with new settings."""
    try:
        processing_settings = None
        try:
            body = await request.json()
            processing_settings = body.get("processing_settings")
        except Exception:
            processing_settings = None
        result = await reframe_client.post(f"/reprocess/{photo_id}", json={"processing_settings": processing_settings})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_system_status():
    """Get system status information."""
    try:
        status = await reframe_client.get("/status")
        return status
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Hardware service unavailable: {str(e)}")


@app.get("/api/health")
async def get_health():
    """Return useful diagnostics even when the camera service is unavailable."""
    battery_task = asyncio.create_task(asyncio.to_thread(read_battery_level))
    hardware_status = None
    try:
        hardware_status = await asyncio.wait_for(
            reframe_client.get("/status"),
            timeout=3,
        )
    except Exception:
        pass

    disk = shutil.disk_usage(PHOTOS_PATH)
    storage = {
        "total": disk.total,
        "used": disk.used,
        "free": disk.free,
    }
    battery = await battery_task
    return build_health_report(
        hardware_status,
        load_diagnostics(),
        storage,
        battery,
    )


@app.get("/api/diagnostics/logs", response_class=PlainTextResponse)
async def get_diagnostic_logs():
    """Load camera logs only when the user opens them in Settings."""
    return await asyncio.to_thread(tail_reframe_log) or "No camera logs recorded yet."


# ═══════════════════════════════════════════════════════════════════
# HARDWARE: Battery — PiSugar 3 via pisugar-server TCP
# Battery level is read by sending "get battery" to pisugar-server
# on TCP port 8423. pisugar-server must be installed separately.
# To use a different battery monitor, replace this endpoint.
# See: https://github.com/PiSugar/PiSugar/wiki/PiSugar-Power-Manager-(Software)
# ═══════════════════════════════════════════════════════════════════
def read_battery_level():
    """Read the PiSugar battery without blocking the dashboard event loop."""
    import subprocess

    try:
        result = subprocess.run(
            ['nc', '-w', '2', '-q', '0', '127.0.0.1', '8423'],
            input='get battery\n',
            capture_output=True,
            text=True,
            timeout=3
        )
        if result.returncode == 0:
            response = result.stdout.strip()
            if response.startswith('battery:'):
                # Handle decimal values and leading spaces
                battery_str = response.split(':')[1].strip()
                battery_level = int(float(battery_str))  # Convert decimal to int
                return {"battery_level": battery_level, "source": "tcp"}

        return {"battery_level": None, "source": "unknown", "error": "Could not connect to PiSugar"}

    except Exception as e:
        return {"battery_level": None, "source": "unknown", "error": str(e)}


@app.get("/api/battery")
async def get_battery_level():
    """Get battery level from PiSugar."""
    return await asyncio.to_thread(read_battery_level)

@app.post("/api/timeout/reset")
async def reset_timeout():
    """Reset the timeout timer (extend the timeout period)."""
    try:
        result = await reframe_client.post("/timeout/reset")
        return result
    except Exception as e:
        logging.info(f"Hardware timeout reset unavailable: {e}")
        return {
            "status": "unavailable",
            "message": "Hardware service is not ready yet"
        }

@app.get("/api/timeout/status")
async def get_timeout_status():
    """Get current timeout status and remaining time."""
    try:
        result = await reframe_client.get("/timeout/status")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get timeout status: {str(e)}")

@app.post("/api/photos/{photo_id}/reprocess")
async def reprocess_single_photo(photo_id: str):
    """Reprocess a single photo to create missing dithered version."""
    try:
        result = await reframe_client.post(f"/reprocess/{photo_id}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reprocess photo: {str(e)}")

# Global variable to track download progress
download_progress = {"status": "idle", "processed": 0, "total": 0, "message": ""}
download_abort = False
download_job_active = False

# Global variable to track delete progress
delete_progress = {"status": "idle", "processed": 0, "total": 0, "message": ""}
delete_abort = False

@app.post("/api/photos/download-all/start")
async def start_download_all(background_tasks: BackgroundTasks):
    """Start the download process and return immediately."""
    global download_progress, download_job_active
    
    if download_job_active or download_progress.get("status") in {"preparing", "creating", "completed", "downloading"}:
        raise HTTPException(status_code=409, detail="A photo download is already in progress")

    try:
        # Get all photos from hardware service
        all_photos = await reframe_client.get("/photos")
        
        if not all_photos:
            raise HTTPException(status_code=404, detail="No photos found")
        
        # Initialize progress
        global download_abort
        download_abort = False
        download_job_active = True
        download_progress = {
            "status": "preparing",
            "processed": 0,
            "total": len(all_photos),
            "message": "Preparing download..."
        }
        
        # Start background task
        background_tasks.add_task(create_zip_background, all_photos)
        
        return {"status": "started", "total_photos": len(all_photos)}
        
    except HTTPException:
        raise
    except Exception as e:
        download_job_active = False
        download_progress = {"status": "error", "processed": 0, "total": 0, "message": str(e)}
        raise HTTPException(status_code=500, detail=f"Failed to start download: {str(e)}")

@app.get("/api/photos/download-all/progress")
async def get_download_progress():
    """Get current download progress."""
    global download_progress
    return download_progress

@app.post("/api/photos/download-all/abort")
async def abort_download():
    """Abort the current download process."""
    global download_abort, download_progress
    download_abort = True
    zip_path = download_progress.get("zip_path")
    if download_progress.get("status") == "completed" and zip_path:
        try:
            os.unlink(zip_path)
        except FileNotFoundError:
            pass
    download_progress = {
        "status": "aborted",
        "processed": download_progress.get("processed", 0),
        "total": download_progress.get("total", 0),
        "message": "Download aborted by user"
    }
    return {"status": "aborted", "message": "Download aborted"}

@app.get("/api/photos/download-all/result")
async def get_download_result():
    """Get the completed ZIP file."""
    global download_progress
    
    print(f"Download result requested. Status: {download_progress.get('status')}")
    
    if download_progress["status"] != "completed":
        print(f"Download not completed. Current status: {download_progress.get('status')}")
        raise HTTPException(status_code=400, detail="Download not completed yet")
    
    zip_path = download_progress.get("zip_path")
    print(f"ZIP path: {zip_path}")
    
    if not zip_path or not os.path.exists(zip_path):
        print(f"ZIP file not found at: {zip_path}")
        raise HTTPException(status_code=404, detail="ZIP file not found")
    
    print(f"Starting file stream for: {zip_path}")
    download_progress["status"] = "downloading"

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"reframe-photos-{datetime.now().strftime('%Y%m%d')}.zip",
        background=BackgroundTask(finish_download_archive, zip_path)
    )


def finish_download_archive(zip_path):
    """Remove an archive after the browser finishes or abandons its transfer."""
    global download_progress
    try:
        os.unlink(zip_path)
        print("Temporary ZIP cleaned up after browser download")
    except FileNotFoundError:
        pass
    except Exception as cleanup_error:
        logging.warning(f"Could not clean up downloaded ZIP {zip_path}: {cleanup_error}")
    finally:
        if download_progress.get("status") in {"downloading", "aborted"}:
            download_progress = {"status": "idle", "processed": 0, "total": 0, "message": ""}


def create_zip_file(all_photos, temp_path):
    """Create the archive in a worker thread and report whether it completed."""
    global download_progress, download_abort
    import zipfile

    download_progress["status"] = "creating"
    download_progress["message"] = "Creating ZIP file..."
    total_photos = len(all_photos)

    with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zip_file:
        for processed, photo in enumerate(all_photos, start=1):
            if download_abort:
                return False

            try:
                original_path = photo.get("original_path")
                if original_path and os.path.exists(original_path):
                    zip_file.write(original_path, f"original/{os.path.basename(original_path)}")

                dithered_path = photo.get("dithered_path")
                if dithered_path and os.path.exists(dithered_path):
                    zip_file.write(dithered_path, f"dithered/{os.path.basename(dithered_path)}")
            except Exception as e:
                print(f"Error adding photo {photo.get('id', 'unknown')} to ZIP: {e}")

            if download_abort:
                return False
            download_progress["processed"] = processed
            download_progress["message"] = f"Processing photo {processed}/{total_photos}"

    return not download_abort


def expire_download_archive(zip_path):
    """Remove a completed archive if no browser claims it within ten minutes."""
    global download_progress
    if (
        download_progress.get("status") == "completed"
        and download_progress.get("zip_path") == zip_path
    ):
        try:
            os.unlink(zip_path)
        except FileNotFoundError:
            pass
        except Exception as cleanup_error:
            logging.warning(f"Could not expire temporary ZIP {zip_path}: {cleanup_error}")
            return
        download_progress = {"status": "idle", "processed": 0, "total": 0, "message": ""}


async def create_zip_background(all_photos):
    """Create one ZIP off the event loop and always clean incomplete files."""
    global download_progress, download_job_active
    temp_path = None
    keep_archive = False
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
            temp_path = temp_file.name

        completed = await asyncio.to_thread(create_zip_file, all_photos, temp_path)
        if not completed:
            download_progress["status"] = "aborted"
            download_progress["message"] = "Download aborted by user"
            return

        archive_size = os.path.getsize(temp_path)
        download_progress["status"] = "completed"
        download_progress["message"] = "ZIP file ready for download"
        download_progress["zip_path"] = temp_path
        download_progress["size_bytes"] = archive_size
        keep_archive = True
        expiry_timer = threading.Timer(600, expire_download_archive, args=(temp_path,))
        expiry_timer.daemon = True
        expiry_timer.start()
        print(f"ZIP creation completed. File: {temp_path}, Size: {archive_size} bytes")
    except Exception as e:
        download_progress["status"] = "error"
        download_progress["message"] = f"Error: {str(e)}"
    finally:
        download_job_active = False
        if temp_path and not keep_archive:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            except Exception as cleanup_error:
                logging.warning(f"Could not remove temporary ZIP {temp_path}: {cleanup_error}")

async def delete_photos_background(all_photos):
    """Background task to delete photos."""
    global delete_progress, delete_abort
    import asyncio
    
    try:
        delete_progress["status"] = "deleting"
        delete_progress["message"] = "Deleting photos..."
        
        total_photos = len(all_photos)
        deleted_count = 0
        
        for photo in all_photos:
            # Check for abort
            if delete_abort:
                delete_progress["status"] = "aborted"
                delete_progress["message"] = "Deletion aborted by user"
                return
            
            try:
                result = await reframe_client.delete(f"/photos/{photo['id']}")
                if result.get("success"):
                    deleted_count += 1
                
                delete_progress["processed"] = deleted_count
                delete_progress["message"] = f"Deleted {deleted_count}/{total_photos} photos"
                
                # Small delay for responsiveness
                if deleted_count % 2 == 0:
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                print(f"Error deleting photo {photo.get('id', 'unknown')}: {e}")
                delete_progress["processed"] = deleted_count
                continue
        
        # Check for abort before marking as completed
        if delete_abort:
            delete_progress["status"] = "aborted"
            delete_progress["message"] = "Deletion aborted by user"
            return
        
        # Mark as completed
        delete_progress["status"] = "completed"
        delete_progress["message"] = f"Successfully deleted {deleted_count} photos"
        
    except Exception as e:
        delete_progress["status"] = "error"
        delete_progress["message"] = f"Error: {str(e)}"

@app.post("/api/photos/delete-all/start")
async def start_delete_all(background_tasks: BackgroundTasks):
    """Start the delete process and return immediately."""
    global delete_progress
    
    try:
        # Get all photos from hardware service
        all_photos = await reframe_client.get("/photos")
        
        if not all_photos:
            return {"status": "completed", "message": "No photos to delete", "deleted_count": 0}
        
        # Initialize progress
        global delete_abort
        delete_abort = False
        delete_progress = {
            "status": "preparing",
            "processed": 0,
            "total": len(all_photos),
            "message": "Preparing deletion..."
        }
        
        # Start background task
        background_tasks.add_task(delete_photos_background, all_photos)
        
        return {"status": "started", "total_photos": len(all_photos)}
        
    except Exception as e:
        delete_progress = {"status": "error", "processed": 0, "total": 0, "message": str(e)}
        raise HTTPException(status_code=500, detail=f"Failed to start deletion: {str(e)}")

@app.get("/api/photos/delete-all/progress")
async def get_delete_progress():
    """Get current delete progress."""
    global delete_progress
    return delete_progress

@app.post("/api/photos/delete-all/abort")
async def abort_delete():
    """Abort the current delete process."""
    global delete_abort, delete_progress
    delete_abort = True
    delete_progress = {
        "status": "aborted",
        "processed": delete_progress.get("processed", 0),
        "total": delete_progress.get("total", 0),
        "message": "Deletion aborted by user"
    }
    return {"status": "aborted", "message": "Deletion aborted"}

@app.post("/api/photos/delete-all")
async def delete_all_photos():
    """Delete all photos from the system."""
    try:
        # Get all photos from hardware service
        all_photos = await reframe_client.get("/photos")
        
        if not all_photos:
            return {"success": True, "message": "No photos to delete"}
        
        deleted_count = 0
        
        # Delete each photo through the hardware service
        for photo in all_photos:
            try:
                result = await reframe_client.delete(f"/photos/{photo['id']}")
                if result.get("success"):
                    deleted_count += 1
            except Exception as e:
                print(f"Error deleting photo {photo.get('id', 'unknown')}: {e}")
                continue
        
        return {
            "success": True, 
            "message": f"Successfully deleted {deleted_count} photos",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete photos: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
