#!/usr/bin/env python3

import os
import sys
import json
import subprocess
import logging
from time import sleep

# OPTIMIZATION: Lazy import PIL
Image = None  # type: ignore
ImageEnhance = None  # type: ignore

def _lazy_import_pil():
    """Import PIL only when needed for image processing."""
    global Image, ImageEnhance
    if Image is None:
        from PIL import Image, ImageEnhance
    return Image, ImageEnhance

from picamera2 import Picamera2
import numpy as np
import smbus2

from typing import Optional, Dict, Any

# OPTIMIZATION: Lazy import FastAPI - saves 4.3 seconds on startup
_API_AVAILABLE = None  # Will be determined when first imported
FastAPI = None  # type: ignore
HTTPException = None  # type: ignore  
Request = None  # type: ignore
uvicorn = None  # type: ignore

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
            FastAPI = None  # type: ignore
            HTTPException = None  # type: ignore
            Request = None  # type: ignore
            uvicorn = None  # type: ignore
    return _API_AVAILABLE

import threading
import time

libdir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib')
if os.path.exists(libdir):
    sys.path.append(libdir)
from waveshare_epd import epd4in0e

# Logging setup - OPTIMIZATION: Reduce logging level to speed up e-ink operations
logging.basicConfig(level=logging.INFO)  # Was DEBUG

# Constants for file paths
BASE_PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_PATH = os.path.join(BASE_PATH, "photos")
PROCESSED_PATH = os.path.join(BASE_PATH, "dithered_photos")

# Color palettes for dithering
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

class CameraManager:
    """Manages camera operations, including capturing and processing photos."""

    def __init__(self, settings_path="settings.json"):
        self.settings_path = settings_path
        self.settings = self.load_settings()
        self.picam2 = Picamera2()
        self.last_activity_time = time.time()
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
                    "display_timeout": 0
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
            file_path = file_manager.get_new_file_path(SAVE_PATH, "png")
        
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
        self.last_activity_time = time.time()
    
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
        elapsed = time.time() - self.last_activity_time
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
        self.update_activity_time()
        # OPTIMIZATION: Faster autofocus for startup photo
        if fast_mode:
            sleep(0.1)  # Fast autofocus for startup
            logging.info("Fast autofocus mode: 0.1s delay")
        else:
            sleep(0.3)  # Normal autofocus delay
        self.picam2.capture_file(file_path)  # Capture the photo
        logging.info(f"Photo saved to {file_path}")

class ImageProcessor:
    """Handles image processing, including resizing, dithering, and saving."""

    @staticmethod
    def get_bayer_matrix(size):
        """Generate Bayer matrix for ordered dithering."""
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
        """Apply optimized ordered dithering using Bayer matrix."""
        # OPTIMIZATION: Lazy import PIL only when doing image processing
        Image, ImageEnhance = _lazy_import_pil()
        
        import time
        start_time = time.time()
        
        # Ensure the image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Adjust brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)

        # Adjust saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(color_factor)

        # Get the blended palette (unique colors in hardware order: K, W, Y, R, B, G)
        palette_colors = []
        color_indices = [0, 1, 5, 4, 0, 3, 2]  # Match Floyd-Steinberg palette structure
        for i in color_indices:
            rs, gs, bs = [c * saturation for c in SATURATED_PALETTE[i]]
            rd, gd, bd = [c * (1.0 - saturation) for c in DESATURATED_PALETTE[i]]
            palette_colors.append([int(rs + rd), int(gs + gd), int(bs + bd)])

        # Convert image to numpy array
        img_array = np.array(image, dtype=np.float32)
        height, width, channels = img_array.shape

        # Get Bayer matrix and tile it to cover the entire image
        bayer_matrix = ImageProcessor.get_bayer_matrix(bayer_size)
        
        # Create tiled threshold matrix for the entire image (OPTIMIZATION: vectorized)
        y_tiles = (height + bayer_size - 1) // bayer_size
        x_tiles = (width + bayer_size - 1) // bayer_size
        threshold_matrix = np.tile(bayer_matrix, (y_tiles, x_tiles))[:height, :width]
        threshold_matrix = threshold_matrix * threshold_scale
        
        # Convert palette to numpy array for vectorized operations
        palette_array = np.array(palette_colors, dtype=np.float32)
        
        # OPTIMIZATION: Vectorized distance calculation
        # Reshape image for broadcasting: (H*W, 3)
        pixels_flat = img_array.reshape(-1, 3)
        threshold_flat = threshold_matrix.flatten()
        
        # Calculate base distances to all palette colors without threshold
        # pixels_flat: (H*W, 3), palette_array: (N_colors, 3) -> distances: (H*W, N_colors)
        base_distances = np.sum((pixels_flat[:, np.newaxis, :] - palette_array[np.newaxis, :, :]) ** 2, axis=2)
        
        # Find the two closest colors for each pixel
        sorted_indices = np.argsort(base_distances, axis=1)
        closest_color1 = sorted_indices[:, 0]  # Closest color
        closest_color2 = sorted_indices[:, 1]  # Second closest color
        
        # FIXED: Proper ordered dithering - use threshold to choose between two closest colors
        # Normalize threshold to [0, 1] range
        normalized_threshold = threshold_flat
        
        # For each pixel, decide between the two closest based on relative distances.
        # Probability of choosing the second color increases as it gets closer to the pixel
        # compared to the first. ratio = d1/(d1 + d2) ∈ (0, 0.5]; choose second if threshold < ratio.
        d1 = base_distances[np.arange(base_distances.shape[0]), closest_color1]
        d2 = base_distances[np.arange(base_distances.shape[0]), closest_color2]
        ratio = d1 / (d1 + d2 + 1e-9)
        use_second_color = normalized_threshold < ratio
        
        # Create final color indices
        closest_indices = np.where(use_second_color, closest_color2, closest_color1)
        
        # Reshape back to image dimensions
        output_array = closest_indices.reshape(height, width).astype(np.uint8)

        # Create palette image
        palette_flat = []
        for color in palette_colors:
            palette_flat.extend(color)
        palette_flat += [0, 0, 0] * (256 - len(palette_colors))

        # Create output image
        output_image = Image.fromarray(output_array, mode='P')
        output_image.putpalette(palette_flat)

        end_time = time.time()
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
            # OPTIMIZATION: Lazy import PIL only when doing image processing
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
    def resize_image(image, size=(600, 400)):
        """Resizes the image to the specified size."""
        # Pillow >= 10 deprecates Image.ANTIALIAS in favor of Image.Resampling.LANCZOS
        resampling_attr = getattr(Image, "Resampling", Image)
        resample_filter = getattr(resampling_attr, "LANCZOS", Image.LANCZOS)
        return image.resize(size, resample_filter)

    @staticmethod
    def img2buffer(image, width=400, height=600):
        """Converts an image to a format suitable for the e-ink display."""
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
            # OPTIMIZATION: Lazy import PIL only when doing image processing
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

    def get_new_file_path(self, folder, extension="png"):
        """Generates a new unique file path in the specified folder."""
        index = len(os.listdir(folder))
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


class EInkDisplay:
    """Manages the e-ink display with lazy initialization for faster startup."""

    def __init__(self):
        # OPTIMIZATION: Don't initialize e-ink hardware immediately - saves 5-10s on startup
        self.epd = None
        self._initialized = False
        logging.info("E-ink display: Lazy initialization enabled")

    def _ensure_initialized(self):
        """Initialize e-ink display only when first needed."""
        if not self._initialized:
            logging.info("Initializing e-ink display hardware...")
            import time
            start_time = time.time()
            
            self.epd = epd4in0e.EPD()
            self.epd.init()
            
            init_time = time.time() - start_time
            logging.info(f"E-ink display ready in {init_time:.2f}s")
            self._initialized = True

    def display_image(self, image):
        """Displays the provided image on the e-ink display."""
        self._ensure_initialized()  # Initialize only when first used
        buffer = ImageProcessor.img2buffer(image)
        if buffer:
            self.epd.display(buffer)

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
            # OPTIMIZATION: Lazy import PIL only when needed for display
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
                if image.size != (600, 400):
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
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to display photo {photo_id}: {str(e)}"
            }

    def clear_display(self):
        """Clear the e-ink display."""
        self._ensure_initialized()  # Initialize only when first used
        try:
            self.epd.Clear()
            logging.info("E-ink display cleared")
            return {"success": True, "message": "Display cleared"}
        except Exception as e:
            logging.error(f"Error clearing display: {e}")
            return {"success": False, "error": str(e)}

    def sleep(self):
        """Puts the e-ink display to sleep."""
        if self._initialized and self.epd:
            self.epd.sleep()


class CameraSystem:
    """Complete camera system that implements dashboard-like functionality."""
    
    def __init__(self, settings_path="settings.json"):
        self.camera_manager = CameraManager(settings_path)
        self.file_manager = FileManager(SAVE_PATH, PROCESSED_PATH)
        self.eink_display = EInkDisplay()
        self.timeout_thread = None
        self.timeout_running = False
        self._timeout_started = False
        logging.info("Timeout monitor initialization deferred for fast startup")
    
    def start_timeout_monitor(self):
        """Start the background timeout monitoring thread."""
        if not self._timeout_started and self.camera_manager.is_timeout_enabled():
            self.timeout_running = True
            self.timeout_thread = threading.Thread(target=self._timeout_monitor_loop, daemon=True)
            self.timeout_thread.start()
            self._timeout_started = True
            timeout_minutes = self.camera_manager.get_timeout_minutes()
            current_time = time.time()
            last_activity = self.camera_manager.last_activity_time
            elapsed = current_time - last_activity
            logging.info(f"Auto-timeout monitor started: {timeout_minutes} minutes timeout, last activity was {elapsed:.1f}s ago")
    
    def start_timeout_monitor_deferred(self):
        """Start the timeout monitor after first photo for faster startup."""
        if not self._timeout_started:
            logging.info("Starting deferred timeout monitor...")
            self.start_timeout_monitor()
    
    def stop_timeout_monitor(self):
        """Stop the background timeout monitoring thread."""
        self.timeout_running = False
        if self.timeout_thread and self.timeout_thread.is_alive():
            self.timeout_thread.join(timeout=1)
    
    def _timeout_monitor_loop(self):
        """Background loop that checks for timeout and shuts down system."""
        # Wait a bit before starting to check for timeout to avoid false triggers on startup
        time.sleep(60)  # Wait 1 minute before first timeout check
        
        while self.timeout_running:
            try:
                if self.camera_manager.is_timeout_exceeded():
                    logging.info("Timeout exceeded, initiating system shutdown")
                    self.camera_manager.shutdown_system()
                    # If we reach here, shutdown failed, so stop monitoring
                    break
                
                # Check every 30 seconds
                for _ in range(30):
                    if not self.timeout_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logging.error(f"Error in timeout monitor: {e}")
                time.sleep(10)  # Wait before retrying
    
    def update_activity(self):
        """Update activity time."""
        self.camera_manager.update_activity_time()
        
    def capture_photo_api(self, fast_mode=False, ultra_fast_startup=False):
        """API-style photo capture that returns metadata."""
        try:
            photo_path = self.file_manager.get_new_file_path(SAVE_PATH, "png")
            logging.info(f"Capturing photo to: {photo_path}")
            
            result = self.camera_manager.capture_photo_with_metadata(photo_path, fast_mode=fast_mode)
            
            if result["success"]:
                logging.info(f"Photo captured successfully: {result['photo_id']}")
                
                processing_settings = self.camera_manager.settings.get("processing", {})
                dithered_path = os.path.join(PROCESSED_PATH, f"{result['photo_id']}_dithered.png")
                logging.info(f"Processing to: {dithered_path}")
                
                process_result = ImageProcessor.process_photo_with_settings(
                    photo_path, dithered_path, processing_settings
                )
                
                if process_result["success"]:
                    result["processed_path"] = dithered_path
                    logging.info(f"Dithered version created: {dithered_path}")
                    
                    display_settings = self.camera_manager.settings.get("display", {})
                    if display_settings.get("auto_display", True):
                        logging.info("Auto-displaying photo")
                        self.eink_display.display_photo_by_id(result["photo_id"], self.file_manager)
                        
                else:
                    logging.error(f"Processing failed: {process_result.get('error', 'unknown error')}")
                
            return result
            
        except Exception as e:
            logging.error(f"Error in capture_photo_api: {e}")
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
                "display_active": True
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
        with _operation_lock:
            camera_system.update_activity()
            result = camera_system.capture_photo_api()
        return result

    @app.post("/api/display/{photo_id}")
    def api_display(photo_id: str):
        global camera_system
        if camera_system is None:
            raise HTTPException(status_code=503, detail="Camera system not initialized")
        with _operation_lock:
            camera_system.update_activity()
            result = camera_system.display_photo_api(photo_id)
        return result

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
        return camera_system.list_photos_api()

    @app.get("/api/photos/{photo_id}")
    def api_get_photo(photo_id: str):
        global camera_system
        if camera_system is None:
            raise HTTPException(status_code=503, detail="Camera system not initialized")
        info = camera_system.get_photo_info_api(photo_id)
        if not info:
            raise HTTPException(status_code=404, detail="Photo not found")
        return info

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
        return camera_system.get_system_status_api()
    
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
            elapsed = time.time() - camera_system.camera_manager.last_activity_time
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
            "last_activity": camera_system.camera_manager.last_activity_time
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
    # OPTIMIZATION: Lazy import FastAPI only when API server starts (after first photo)
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
    
    camera_system = CameraSystem()
    logging.info("📷 Camera system initialized")
    logging.info("📸 Taking startup photo...")
    try:
        with _operation_lock:
            result = camera_system.capture_photo_api(fast_mode=True, ultra_fast_startup=True)
        
        if result.get("success"):
            logging.info("✅ FAST startup photo captured: %s", result.get("photo_id", "unknown"))
            if camera_system.camera_manager.settings.get("display", {}).get("auto_display", True):
                logging.info("🖥️  Startup photo displayed on screen")
            logging.info("🏁 FAST SYSTEM READY")
            
            # Ensure activity time is updated before starting timeout monitor
            camera_system.update_activity()
            camera_system.start_timeout_monitor_deferred()
        else:
            logging.warning("❌ Failed to capture startup photo: %s", result.get("message", "unknown error"))
    except Exception as e:
        logging.error("💥 Error taking startup photo: %s", e)

    # Initialize I2C for power button detection
    I2C_ADDRESS = 0x57
    BUTTON_REGISTER = 0x02
    bus = smbus2.SMBus(1)

    def is_power_button_pressed():
        try:
            reg_val = bus.read_byte_data(I2C_ADDRESS, BUTTON_REGISTER)
            return bool(reg_val & 0x01)  # Check the least significant bit
        except Exception as e:
            logging.error("Failed to read I2C: %s", e)
            return False

    prev_state = False

    # Start API server in background
    try:
        port = int(os.environ.get("REFRAME_API_PORT", "8077"))
    except Exception:
        port = 8077
    _start_api_server_in_background(host="127.0.0.1", port=port)

    logging.info("System initialized. API server running. Waiting for button press to capture photo...")

    try:
        while True:
            current_state = is_power_button_pressed()
            if current_state and not prev_state:
                logging.info("Button pressed. Capturing photo...")
                with _operation_lock:
                    result = camera_system.capture_photo_api()
                if result.get("success"):
                    logging.info("Photo captured%s.", " and displayed" if camera_system.camera_manager.settings.get("display", {}).get("auto_display", True) else "")
                else:
                    logging.error("Capture failed: %s", result.get("message", "unknown error"))

                # Pause briefly to debounce and allow the user to view the image
                sleep(2)
            prev_state = current_state
            sleep(0.1)
    except KeyboardInterrupt:
        logging.info("Program interrupted by user. Exiting...")
    finally:
        bus.close()
        try:
            # Stop timeout monitor and put display to sleep if initialized inside CameraSystem
            if camera_system:
                camera_system.stop_timeout_monitor()
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
