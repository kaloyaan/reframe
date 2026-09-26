# Build Guide

Everything you need to build your own reFrame camera. This version intentionally uses only off-the-shelf components so it's easy to customize.

The build requires some 3D printing and soldering experience, so it's not necessarily for absolute beginners. If you need help with any part of the process, you can reach out via email or Instagram.

## Materials

Approximate cost: **~$200-250 USD per camera**

| # | Component | Part | ~Price | Link | Notes |
|---|-----------|------|--------|------|-------|
| 1 | Computer | Raspberry Pi Zero 2 WH | $20 | [PiShop.us](https://www.pishop.us/product/raspberry-pi-zero-2w-with-headers/) | WH version has pre-soldered headers for easier assembly. Non-header version works as well. |
| 2 | Camera | Raspberry Pi Camera Module 3 | $25 | [Digi-Key](https://www.digikey.com/en/products/detail/raspberry-pi/SC1223/17278639) | Standard field-of-view model. The Camera Module 3 is recommended because the autofocus and HDR make a big difference in image quality on the ePaper display. |
| 3 | Display | Waveshare 4" ePaper Spectra 6 (HAT+) | $50 | [Waveshare](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?sku=27367) | Connects directly to the Pi's GPIO header — no additional wiring needed. The [driver](../waveshare_epd/) is bundled in this repo. |
| 4 | Battery | PiSugar 3 Battery Board | $40 | [PiSugar](https://www.pisugar.com/products/pisugar-3-raspberry-pi-zero-battery) or [Amazon](https://www.amazon.com/dp/B0FB3N1YSK) | Provides power and physical button interface. This build works only with PiSugar 3 because of how it handles on-off buttons. Requires [`pisugar-server`](https://github.com/PiSugar/PiSugar/wiki/PiSugar-Power-Manager-(Software)) for monitoring; the [Software Setup](software-setup.md) guide installs it. |
| 5 | Storage | 32 GB microSD card | $10 | Any brand, the faster, the better. | For OS and photo storage. |
| 6 | Push Button | 12mm Stainless Steel Push Button | $6 | [Amazon](https://www.amazon.com/dp/B0811QKG1R) | Used as a combined hardware on/off + shutter button. |
| 7 | Camera Cable | Short Pi ribbon cable | $6 | [Amazon](https://www.amazon.com/dp/B085RW9K13?th=1) | For neatness; can also use the one that comes with the camera. |
| 8 | Nylon Standoffs | M2.5 Hex Standoff Spacers Kit | $10 | [Amazon](https://www.amazon.com/COMRUN-M2-5-Standoff-Assortment-Motherboard/dp/B0CKBWQSNY/) | Hold the build together. The camera needs 4 20mm standofs; could be nylon or metal. |
| 9 | Metal Screws | M2.5 metal screws | $10 | [Amazon](https://www.amazon.com/dp/B01N5RDAUX) | Holds the front lid + the USB-C board. Could be metal or nylon (if you want to use the ones from the link above). I went with metal for a nicer look on the front lid.  |
| 10 | Camera Screws | M2 screws | $7 | [Amazon](https://www.amazon.com/QOOSIKICC-Assortment-Washers-Machine-Assorted/dp/B0DDY2L533/) | For mounting the camera module to the enclosure. |
| 11 | USB C Port | USB C Breakout Board | $8 | [Amazon](https://www.amazon.com/Breakout-Connector-Serial-Female-Adapter/dp/B0F2S1K5X8) | Used for external USB-C port access. |
| 12 | Connectors | JST 1.25mm 2-Pin Micro Connectors | $7 | [Amazon](https://www.amazon.com/dp/B013JRWCBU) | So it's easier to connect the USB-C port + power button during assembly. |
| 13 | Filament | Transparent PLA Filament | $20 | [Amazon](https://www.amazon.com/dp/B08VRM8KZJ) | Good for reliable overhangs and fine details (PETG could also work but is more finicky). Translucent so status lights are visible under the board. The one used in my build is the GIANTARM Clear PLA Filament. |
| 14 | Wrist Strap | Wrist strap | $6 | [Amazon](https://www.amazon.com/dp/B0BC8G4ZR2) | Hand strap for carrying. |
| 15 | Putty | Sticky putty | $4 | Any brand | For holding the battery securely in place. |
| 16 | Carry Case | Optional carry case | $12 | [Amazon](https://www.amazon.com/dp/B0DBLMKYGB) | For travel protection. This size (5.5" × 3.7" × 2.8") fits the camera well. |
| 17 | Charging Cable | USB-A to USB-C | - | - | For charging the camera. C-to-C cables might not work unless you solder some extra ports on the USB-C board. |


## Enclosure

The 3D-printable files for the camera case are located in the [hardware/](../hardware/) directory. The model consists of four separate parts:

- **[enclosure.stl](../hardware/enclosure.stl)**: The main body that holds the Pi, battery, and display. It has cutouts for the display, the USB-C port, the power/shutter button, and the wrist strap.
- **[lid.stl](../hardware/lid.stl)**: The back cover of the camera. It attaches to the main enclosure using 4x M2.5 screws that screw into the standoffs integrated into the main enclosure.
- **[bezel.stl](../hardware/bezel.stl)**: The front bezel that frames the ePaper display. It attaches to the notches of the enclosure.
- **[ring.stl](../hardware/ring.stl)**: The accent ring that protects the camera lens.

For modification or custom adjustments, you can also download:
- **[reframe.f3d](../hardware/reframe.f3d)**: The Fusion 360 CAD source file.
- **[reframe.step](../hardware/reframe.step)**: The STEP file for use in other CAD software.

If you're using BambuStudio as your slicer:
- **[reframe-3dprint.3mf](../hardware/reframe-3dprint.3mf)**: A pre-configured BambuLab project file containing all parts and settings.

### Slicer Settings

For optimal printing, use the following slicer settings:

![Slicer Layout](images/build-guide/slicer-settings.png)
*Recommended print layout and orientation for all four components*

![Slicer Parameters](images/build-guide/slicer-settings-2.png)
*Supports needed for the overhangs of the enclosure.*


## 3D Printed Parts & Tools

Before starting assembly, make sure you have printed all the necessary enclosure parts and gathered the required tools.

![3D Printed Parts](images/build-guide/001-3D-printed-parts.jpg)
*The 3D printed parts needed for the enclosure, printed in transparent PLA – missing here is the front panel.*

### Helpful Tools

![Tools Needed](images/build-guide/002-tools.jpg)

- 3D printer (or 3D printing service)
- Soldering iron and solder
- Small Phillips screwdriver
- microSD card reader (for your computer)
- A computer with [Raspberry Pi Imager](https://www.raspberrypi.com/software/) installed
- Tweezers or pliers
- Sticky putty (for holding the battery in place)
- Wire stripper and cutter
- Heatshrink tubing or electrical tape

## Step 1: Prepare the SD Card

1. Insert the microSD card into your computer.
2. Open Raspberry Pi Imager.
3. Select the settings:
   - **Device**: Raspberry Pi Zero 2 W
   - **OS**: Raspberry Pi OS (other) > Raspberry Pi OS Lite (64-bit)
   - **Hostname**: `reframe`
   - **Username**: `cam`
   - **Password**: your choice (but write it down somewhere safe)
   - **WiFi**: enter your home network name and password. If you want to use the camera on the go, you can add your phone's hotspot.
   - **Services**: Enable SSH with password authentication
4. Write the image and eject the card.

The username `cam` is required by the current service and updater configuration.

## Step 2: Soldering & Wiring Preparations

In this step, we will prepare the wire connections for the shutter button, the PiSugar 3 battery board, and the USB-C breakout board.

![PiSugar 3 Solder Pad Diagram](images/build-guide/reframe-wiring.svg)
*Simplified PiSugar 3 soldering points for this build. For the button connector, solder black to Pad 10 (Power Button Pad) and red to BAT+ / Pad 2. For the USB-C connector, solder black to Pad 1 (GND) and red to Pad 8 (5V Input Pad).*

### A. Soldering the Shutter Button
Prepare the 12mm stainless steel push button by soldering wires to its terminals.

![Soldering Button Wires](images/build-guide/003-soldering-button-wires.jpg)
*Tin the leads and the terminals of the push button before soldering.*

![Soldering Button Contacts](images/build-guide/004-soldering-button-contacts.jpg)
*Securely solder the wires to the contacts. Use heat shrink or electrical tape to prevent shorts.*

### B. Soldering the USB-C Breakout Board
Prepare the USB-C breakout board to expose the charging port through the enclosure.

![USB-C Breakout Prepared](images/build-guide/011-usb-c-breakout.jpg)
*Solder the power wires to the VBUS and GND pads on the breakout board.*

### C. Soldering the PiSugar 3 Battery Board
Take two JST 1.25mm 2-Pin Micro Connectors. Connect the two wires of each connector to the pads on the PiSugar 3 board. You need to connect two connectors - one for the power button, and one for the USB-C charging board.

![PiSugar 3 Solder Pad Diagram](images/build-guide/reframe-wiring-closeup.svg)

For a detailed PiSugar PCB diagram, see the [PiSugar 3 wiki](https://github.com/PiSugar/PiSugar/wiki/PiSugar-3-Series).

| Position | Name | Description |
|:---:|---|---|
| **1** | GND | There are several ground pins you can use for this. |
| **2** | BAT | The battery should be soldered on there already, just add an extra solder on top. |
| **8** | 5V Input Pad | Connects to our external USB board |
| **10** | Power Button Pad | Used to trigger the power button when connected to Pad 2. |


![Soldering PiSugar Done](images/build-guide/007-soldering-on-pisugar-done.jpg)
*Carefully solder the power button wires. Sticky putty helps hold the board in place. Watch out for the magnet.*

![Soldering PiSugar Wires](images/build-guide/005-soldering-on-pisugar-wires.jpg)
*Completed soldering on the PiSugar 3 battery board.*

### D. Bench Test the Power Wiring
Before putting everything into the enclosure, test the soldered button and charging connections while the PiSugar board is still easy to reach.

1. Plug the external button JST connector into the PiSugar button harness.
2. Press the button and confirm the PiSugar reacts. On a new PiSugar 3, accidental-touch prevention may still be enabled, so power-on may require a short press followed by a long press.
3. Plug a USB-A to USB-C cable into the external USB-C breakout board.
4. Confirm the PiSugar charging LEDs turn on.
5. Unplug USB-C and power the board off before continuing the mechanical assembly.

If either test fails, stop and recheck the pad wiring before mounting the stack in the case.


## Step 3: Component Mounting & Enclosure Prep

In this step, we prepare the ePaper display and mount the camera, button, and wrist strap into the 3D-printed enclosure.

### A. Preparing the Display
Unbox the Waveshare 4" ePaper display HAT+ and prepare it for mounting. Screw on the included standoffs – you only need the two next to the header.

![ePaper Display Front](images/build-guide/006-display-s.jpg)
*The back of the Waveshare Spectra 6 ePaper display. The 40-pin header can't be removed.*

Insert the display into the holes of the 3D printed enclosure, and secure it with 4 20mm M2.5 standoffs.

![ePaper Display Back View 1](images/build-guide/008-display-x.jpg)
*Rear side of the display showing the interface connectors.*

Once the display is secured, you can snap on the protective bezel.

![ePaper Display Back View 2](images/build-guide/009-display-y.jpg)
*Ignore the wrist strap in this image, we're adding it later.*

Just align the holes and push, if the tolerance is right it should snap in.

![ePaper Display Back View 3](images/build-guide/010-display-z2.jpg)

### B. Mounting the USB-C Board

Mount the USB-C breakout board to the case and screw it in using the M2.5 screws.

![USB-C Board Mounted](images/build-guide/012-usb-c-wires.jpg)

### C. Mounting the Shutter Button & Wrist Strap
Install the wrist strap by sliding it in through the outside hole, then positioning the ear so it holds the spacer.

Slide the button into the 3D-printed shell and secure it from the other side using the ring it came with.

![Button Mounted](images/build-guide/013-button-mount.jpg)
*Screw the 12mm stainless steel push button into its designated slot in the enclosure.*

![Wrist Strap Mounted](images/build-guide/014-wrist-strap.jpg)
*Thread the wrist strap loop through the anchor holes in the case.*

### D. Mounting the Camera Module 3
Mount the Raspberry Pi Camera Module 3 to the front plate.

![Camera Alignment](images/build-guide/015-camera-mount.jpg)
*Align the camera module lens with the front plate cutout.*

![Camera Screws Back](images/build-guide/016-camera-back.jpg)
*Secure the camera module using the M2 screws (one's missing from this photo). Ignore the ribbon cable for now, we'll snap it in later.*

The screws should friction fit into the protective front ring.

![Camera Mounted Front View](images/build-guide/017-camera-ready.jpg)
*The camera module securely mounted to the front faceplate.*


## Step 4: Assembling the Electronics Stack

Now, we assemble the main compute stack by pairing the Raspberry Pi Zero 2 W with the PiSugar 3 battery board and connecting the camera.

### A. Mating the Pi and PiSugar
Mount the Raspberry Pi Zero 2 W directly onto the PiSugar 3 battery board.

![Pi on PiSugar](images/build-guide/018-pi-on-top-of-pisugar.jpg)
*Place the Pi Zero 2 W on top of the PiSugar 3. The pogo pins will contact the power pads on the back of the Pi.*

![Pi Stack Secured](images/build-guide/019-pi-covered.jpg)
*Secure the Pi to the PiSugar using the M2.5 screws. Covering the back of the PiSugar with electrical tape helps prevent shorts.*

### B. Connecting the Camera Ribbon Cable
Connect the camera ribbon cable to the Pi Zero.

![Connect Camera Cable](images/build-guide/020-connect-camera-to-pi.jpg)
*Lift the latch on the Pi Zero camera port, slide the ribbon cable in, and clamp it down securely.*


### C. Insert the SD Card
Insert your flashed SD Card into the Pi Zero 2 W.


## Step 5: Final Enclosure Assembly

Fit all the sub-assemblies together inside the enclosure and wire up the remaining connections.

### A. Fitting the stack in the Case
Position the main electronics stack inside the 3D-printed enclosure. The Pi 40-pin header should fit into the header on the display.

![Mount Stack in Case](images/build-guide/021-mount-in-case.jpg)
*Carefully position the Pi/battery stack into the bottom shell.*

### B. Connecting Wires
Attach the shutter button and USB-C breakout board connectors to the main board.

![Connect Button and USB Wires](images/build-guide/022-connect-button-usb.jpg)
*Plug the JST 1.25mm connectors from the button and USB-C breakout board into the Pi/PiSugar headers.*

Remove the battery from the magnet and lay it flat next to the board. You can use some sticky putty to keep it in place so it doesn't wiggle.

### C. Screwing the Case Together
Secure the front cover plates using the four M2.5 screws.

![Screws on Top](images/build-guide/023-screws-on-top.jpg)
*Insert the M2.5 screws into the corner holes to fasten the enclosure assembly.*

## Step 6: Software Setup & First Boot

With assembly complete, configure the software and boot up your reFrame camera. The first boot may still use the PiSugar 3's default accidental-touch prevention behavior, so use the short-press-then-long-press sequence if a single press does not power it on yet.

1. Power on the camera. On a new PiSugar 3, this may require a short press followed by a long press until the software setup disables accidental-touch prevention.
2. Follow the [Software Setup](software-setup.md) guide. During setup, reFrame uses PiSugar Power Manager to disable accidental-touch prevention so the camera powers on with a single press.
3. After setup, press once to power on. reFrame automatically takes and displays a photo when it boots.
4. While the camera is on, short-press the button to take another photo.
5. Long-press the button to shut the camera down.
6. If you leave the camera on, it automatically shuts down after 10 minutes of inactivity to prevent battery drain.

![First Boot](images/build-guide/024-power-on.jpg)
*Powering on the camera. The ePaper screen will cycle colors and refresh to display its very first capture.*

*Need help? Reach out via email or on [Instagram](https://instagram.com/reframe.camera).*
