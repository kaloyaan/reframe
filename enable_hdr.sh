#!/bin/bash

# Enable HDR (Wide Dynamic Range) for Raspberry Pi Camera Module 3
# This script sets the wide_dynamic_range control via v4l2-ctl

# Wait for camera devices to be ready
sleep 2

# Find the correct camera subdevice
for dev in /dev/v4l-subdev*; do
    if [ -e "$dev" ]; then
        # Try to set HDR on this device
        if v4l2-ctl --set-ctrl wide_dynamic_range=1 -d "$dev" 2>/dev/null; then
            echo "HDR enabled on $dev"
            logger "Reframe: HDR enabled on $dev"
            break
        fi
    fi
done

# Also try the main video device as fallback
for dev in /dev/video*; do
    if [ -e "$dev" ]; then
        if v4l2-ctl --set-ctrl wide_dynamic_range=1 -d "$dev" 2>/dev/null; then
            echo "HDR enabled on $dev (video device)"
            logger "Reframe: HDR enabled on $dev (video device)"
            break
        fi
    fi
done

