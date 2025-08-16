#!/usr/bin/env python3

import os
import sys
import json
import subprocess
import logging
from time import sleep
from PIL import Image, ImageEnhance
from picamera2 import Picamera2
import numpy as np
import smbus2

libdir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib')
if os.path.exists(libdir):
    sys.path.append(libdir)
from waveshare_epd import epd4in0e

# Logging setup
logging.basicConfig(level=logging.DEBUG)

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
                    "exposure_value": -0.25,
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

    def capture_photo_with_metadata(self, file_path=None):
        """Capture a photo and return metadata like the dashboard API."""
        if file_path is None:
            # Generate a new file path
            import time
            timestamp = str(int(time.time()))
            file_path = os.path.join(SAVE_PATH, f"{timestamp}.jpg")
        
        try:
            self.capture_photo(file_path)
            
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

    def enable_hdr_hardware(self):
        """Enable HDR via v4l2-ctl command as a fallback/supplement to Picamera2 controls."""
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            hdr_script = os.path.join(script_dir, "enable_hdr.sh")
            
            # Try to enable HDR on camera subdevices
            subprocess.run([hdr_script], check=False, capture_output=True)
            logging.info("HDR hardware script executed")
        except Exception as e:
            logging.warning(f"Could not execute HDR script: {e}")

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
            "ExposureValue": camera_settings.get("exposure_value", -0.25),
            "Sharpness": camera_settings.get("sharpness", 3)
        }
        
        # Always enable HDR via hardware method for Camera Module 3
        self.enable_hdr_hardware()
        
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

    def capture_photo(self, file_path):
        """Capture a photo and save it to the specified file path."""
        sleep(1)  # Allow autofocus to complete
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
        """Apply ordered dithering using Bayer matrix."""
        # Ensure the image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Adjust brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)

        # Adjust saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(color_factor)

        # Get the blended palette
        palette_colors = []
        color_indices = [0, 1, 5, 4, 0, 3, 2]
        for i in color_indices:
            rs, gs, bs = [c * saturation for c in SATURATED_PALETTE[i]]
            rd, gd, bd = [c * (1.0 - saturation) for c in DESATURATED_PALETTE[i]]
            palette_colors.append([int(rs + rd), int(gs + gd), int(bs + bd)])

        # Convert image to numpy array
        img_array = np.array(image, dtype=np.float32)
        height, width, channels = img_array.shape

        # Get Bayer matrix
        bayer_matrix = ImageProcessor.get_bayer_matrix(bayer_size)
        
        # Create output array
        output_array = np.zeros((height, width), dtype=np.uint8)

        # Apply ordered dithering
        for y in range(height):
            for x in range(width):
                # Get pixel values
                pixel = img_array[y, x]
                
                # Get threshold from Bayer matrix
                threshold = bayer_matrix[y % bayer_size, x % bayer_size] * threshold_scale * 255
                
                # Find closest palette colors
                min_distance = float('inf')
                closest_index = 0
                
                for i, palette_color in enumerate(palette_colors):
                    # Calculate distance with dithering threshold
                    adjusted_pixel = pixel + (threshold - 127.5)
                    distance = np.sum((adjusted_pixel - palette_color) ** 2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_index = i
                
                output_array[y, x] = closest_index

        # Create palette image
        palette_flat = []
        for color in palette_colors:
            palette_flat.extend(color)
        palette_flat += [0, 0, 0] * (256 - len(palette_colors))

        # Create output image
        output_image = Image.fromarray(output_array, mode='P')
        output_image.putpalette(palette_flat)

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

        buf_6color = np.frombuffer(image_temp.tobytes('raw'), dtype=np.uint8)
        buf = (buf_6color[0::2] << 4) + buf_6color[1::2]
        buf = buf.astype(np.uint8).tolist()

        return buf

    @staticmethod
    def process_photo_with_settings(original_path, output_path, processing_settings):
        """Process a photo with specific settings and save it."""
        try:
            from PIL import Image
            
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
            
            # Save processed image
            dithered_image.save(output_path, format="JPEG")
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
        for ext in ['jpg', 'jpeg', 'png']:
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
        output_file_path = os.path.join(output_path, f"{photo_id}_dithered.jpg")
        
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

    def save_image(self, image, folder, extension="jpg"):
        """Saves the image to a unique file in the specified folder."""
        file_path = self.get_new_file_path(folder, extension)
        # Choose image format based on extension
        ext_lower = extension.lower()
        if ext_lower in ("jpg", "jpeg"):
            save_format = "JPEG"
        elif ext_lower == "png":
            save_format = "PNG"
        else:
            save_format = "PNG"
        image.save(file_path, format=save_format)
        logging.info(f"Image saved to {file_path}")
        return file_path

    def get_photo_info(self, photo_id):
        """Get information about a specific photo by ID."""
        # Find original photo
        original_path = None
        for ext in ['jpg', 'jpeg', 'png']:
            test_path = os.path.join(self.save_path, f"{photo_id}.{ext}")
            if os.path.exists(test_path):
                original_path = test_path
                break
        
        if not original_path:
            return None
        
        # Check for dithered version
        dithered_path = os.path.join(self.processed_path, f"{photo_id}_dithered.jpg")
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
            photo_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
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
        for ext in ['jpg', 'jpeg', 'png']:
            original_path = os.path.join(self.save_path, f"{photo_id}.{ext}")
            if os.path.exists(original_path):
                os.remove(original_path)
                deleted_files.append(original_path)
                break
        
        # Delete processed version
        dithered_path = os.path.join(self.processed_path, f"{photo_id}_dithered.jpg")
        if os.path.exists(dithered_path):
            os.remove(dithered_path)
            deleted_files.append(dithered_path)
        
        if deleted_files:
            logging.info(f"Deleted photo {photo_id}: {deleted_files}")
            return {"success": True, "deleted_files": deleted_files}
        else:
            return {"success": False, "error": "Photo not found"}


class EInkDisplay:
    """Manages the e-ink display."""

    def __init__(self):
        self.epd = epd4in0e.EPD()
        self.epd.init()

    def display_image(self, image):
        """Displays the provided image on the e-ink display."""
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
            from PIL import Image
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
        try:
            self.epd.Clear()
            logging.info("E-ink display cleared")
            return {"success": True, "message": "Display cleared"}
        except Exception as e:
            logging.error(f"Error clearing display: {e}")
            return {"success": False, "error": str(e)}

    def sleep(self):
        """Puts the e-ink display to sleep."""
        self.epd.sleep()


class CameraSystem:
    """Complete camera system that implements dashboard-like functionality."""
    
    def __init__(self, settings_path="settings.json"):
        self.camera_manager = CameraManager(settings_path)
        self.file_manager = FileManager(SAVE_PATH, PROCESSED_PATH)
        self.eink_display = EInkDisplay()
        
    def capture_photo_api(self):
        """API-style photo capture that returns metadata."""
        try:
            # Get a new file path
            photo_path = self.file_manager.get_new_file_path(SAVE_PATH, "jpg")
            
            # Capture photo with metadata
            result = self.camera_manager.capture_photo_with_metadata(photo_path)
            
            if result["success"]:
                # Process the image
                processing_settings = self.camera_manager.settings.get("processing", {})
                dithered_path = os.path.join(PROCESSED_PATH, f"{result['photo_id']}_dithered.jpg")
                
                process_result = ImageProcessor.process_photo_with_settings(
                    photo_path, dithered_path, processing_settings
                )
                
                if process_result["success"]:
                    result["processed_path"] = dithered_path
                    
                    # Auto-display if enabled
                    display_settings = self.camera_manager.settings.get("display", {})
                    if display_settings.get("auto_display", True):
                        self.eink_display.display_photo_by_id(result["photo_id"], self.file_manager)
                
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


def main():
    camera_manager = CameraManager()
    file_manager = FileManager(SAVE_PATH, PROCESSED_PATH)
    eink_display = EInkDisplay()

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

    logging.info("System initialized. Waiting for button press to capture photo...")

    try:
        while True:
            current_state = is_power_button_pressed()
            if current_state and not prev_state:
                logging.info("Button pressed. Capturing photo...")

                # Capture photo
                photo_path = file_manager.get_new_file_path(SAVE_PATH, "jpg")
                camera_manager.capture_photo(photo_path)

                # Load and process the image
                original_image = Image.open(photo_path)
                resized_image = ImageProcessor.resize_image(original_image)
                
                # Get processing settings
                processing_settings = camera_manager.settings.get("processing", {})
                dithering_method = processing_settings.get("dithering_method", "floyd_steinberg")
                saturation = processing_settings.get("saturation", 0.6)
                brightness_factor = processing_settings.get("brightness_factor", 1.1)
                color_factor = processing_settings.get("color_factor", 1.4)
                bayer_size = processing_settings.get("bayer_size", 4)
                threshold_scale = processing_settings.get("threshold_scale", 1.0)
                
                dithered_image = ImageProcessor.apply_dithering(
                    resized_image, saturation, brightness_factor, color_factor,
                    dithering_method, bayer_size, threshold_scale
                )

                # Save processed image
                processed_path = file_manager.save_image(dithered_image, PROCESSED_PATH)

                # Display on e-ink screen
                eink_display.display_image(dithered_image)
                logging.info("Photo captured and displayed.")

                # Pause briefly to debounce and allow the user to view the image
                sleep(2)
            prev_state = current_state
            sleep(0.1)
    except KeyboardInterrupt:
        logging.info("Program interrupted by user. Exiting...")
    finally:
        bus.close()
        eink_display.sleep()


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
