#!/usr/bin/env python3

import os
import sys
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

    def __init__(self):
        self.picam2 = Picamera2()
        self.configure_camera()

    def configure_camera(self):
        """Configure the camera settings."""
        camera_config = self.picam2.create_still_configuration(main={"size": (1200, 800)})
        camera_config["controls"] = {
            "ExposureValue": -0.25,  # EV Compensation
            "Sharpness": 3  # Sharpness level
        }
        self.picam2.configure(camera_config)
        self.picam2.set_controls({"AfMode": 2})  # Set autofocus to continuous mode
        self.picam2.start()

    def capture_photo(self, file_path):
        """Capture a photo and save it to the specified file path."""
        sleep(1)  # Allow autofocus to complete
        self.picam2.capture_file(file_path)  # Capture the photo
        logging.info(f"Photo saved to {file_path}")

class ImageProcessor:
    """Handles image processing, including resizing, dithering, and saving."""

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
    def apply_dithering(image, saturation=0.6, brightness_factor=1.1, color_factor=1.4):
        """Applies brightness, color enhancement, and dithering."""
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
        return image.resize(size, Image.ANTIALIAS)

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
        image.save(file_path, format="PNG")
        logging.info(f"Image saved to {file_path}")
        return file_path


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

    def sleep(self):
        """Puts the e-ink display to sleep."""
        self.epd.sleep()


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
                photo_path = file_manager.get_new_file_path(SAVE_PATH)
                camera_manager.capture_photo(photo_path)

                # Load and process the image
                original_image = Image.open(photo_path)
                resized_image = ImageProcessor.resize_image(original_image)
                dithered_image = ImageProcessor.apply_dithering(resized_image)

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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Program interrupted. Exiting...")
        epd4in0e.epdconfig.module_exit(cleanup=True)
