#!/bin/bash
set -e

# ─────────────────────────────────────────────
#  reFrame installer
#  Run this on a fresh Raspberry Pi OS Lite
# ─────────────────────────────────────────────

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; }
step()  { echo -e "\n${GREEN}── $1 ──${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BOOT_CONFIG="/boot/config.txt"
[ -f "/boot/firmware/config.txt" ] && BOOT_CONFIG="/boot/firmware/config.txt"
BOOT_CMDLINE="/boot/cmdline.txt"
[ -f "/boot/firmware/cmdline.txt" ] && BOOT_CMDLINE="/boot/firmware/cmdline.txt"
REQUIRED_USER="cam"
REQUIRED_REPO="/home/cam/reframe"
CURRENT_USER="$(id -un)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ "$(id -u)" -eq 0 ]; then
    error "Do not run install.sh with sudo or as root."
    error "Log in as '$REQUIRED_USER' and run: cd $REQUIRED_REPO && ./install.sh"
    exit 1
fi

if [ "$CURRENT_USER" != "$REQUIRED_USER" ]; then
    error "reFrame must currently be installed as the '$REQUIRED_USER' user."
    error "The services and software updater use that account."
    error "Re-flash or create the '$REQUIRED_USER' account, then clone the repo to $REQUIRED_REPO."
    exit 1
fi

if [ "$SCRIPT_DIR" != "$REQUIRED_REPO" ]; then
    error "reFrame must currently be cloned at $REQUIRED_REPO."
    error "Current location: $SCRIPT_DIR"
    error "Move or clone the repository to the required location, then run install.sh again."
    exit 1
fi

echo ""
echo "  ┌─────────────────────────────┐"
echo "  │   reFrame camera installer  │"
echo "  └─────────────────────────────┘"
echo ""

# ── 1. System packages ──────────────────────
step "Installing system packages"

sudo apt update -qq
sudo apt install -y -qq \
    git \
    python3-pip \
    python3-venv \
    python3-picamera2 \
    python3-spidev \
    python3-gpiozero \
    v4l-utils \
    i2c-tools \
    netcat-openbsd \
    wget

info "System packages installed"

step "Enabling I2C"

sudo raspi-config nonint do_i2c 0
info "I2C enabled"

# ── 2. Python packages ──────────────────────
step "Installing Python packages"

/usr/bin/python3 -m venv --system-site-packages "$VENV_DIR"
PYTHONNOUSERSITE=1 "$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"

info "Python packages installed in $VENV_DIR"

# ── 3. PiSugar Power Manager ────────────────
step "Checking for PiSugar Power Manager"

configure_pisugar() {
    if ! command -v nc >/dev/null 2>&1; then
        warn "Cannot configure PiSugar automatically: nc is not installed"
        return
    fi

    sudo systemctl start pisugar-server 2>/dev/null || true

    for attempt in 1 2 3 4 5; do
        if echo "get model" | nc -w 2 -q 0 127.0.0.1 8423 >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    if ! echo "get model" | nc -w 2 -q 0 127.0.0.1 8423 >/dev/null 2>&1; then
        warn "pisugar-server is installed but not responding on port 8423"
        warn "After it starts, run: echo 'set_anti_mistouch false' | nc -q 0 127.0.0.1 8423"
        return
    fi

    if echo "set_anti_mistouch false" | nc -w 2 -q 0 127.0.0.1 8423 | grep -qi "done"; then
        info "Disabled PiSugar anti-mistouch so one press powers on the camera"
    else
        warn "Could not disable PiSugar anti-mistouch automatically"
        warn "Manual command: echo 'set_anti_mistouch false' | nc -q 0 127.0.0.1 8423"
    fi
}

install_pisugar_power_manager() {
    local installer="/tmp/pisugar-power-manager.sh"

    info "Installing PiSugar Power Manager"
    wget -q -O "$installer" https://cdn.pisugar.com/release/pisugar-power-manager.sh
    bash "$installer" -c release
}

if systemctl list-unit-files | grep -q pisugar-server; then
    info "pisugar-server is already installed"
    configure_pisugar
else
    warn "pisugar-server not found"
    echo ""
    echo "  The PiSugar 3 requires pisugar-server for battery monitoring,"
    echo "  power management, and one-press power-on configuration."
    echo "  The official PiSugar installer may ask you to select a model."
    echo "  Choose 'PiSugar 3' when prompted."
    echo ""

    install_pisugar_power_manager

    if systemctl list-unit-files | grep -q pisugar-server; then
        configure_pisugar
    else
        warn "PiSugar installer finished, but pisugar-server was not found"
        warn "Manual install: https://github.com/PiSugar/PiSugar/wiki/PiSugar-3-Series"
    fi
fi

# ── 4. Enable hardware interfaces ───────────
step "Enabling hardware interfaces"

# Enable SPI
sudo raspi-config nonint do_spi 0
info "SPI enabled"

if ! grep -q "^camera_auto_detect=1" "$BOOT_CONFIG" 2>/dev/null; then
    echo "camera_auto_detect=1" | sudo tee -a "$BOOT_CONFIG" > /dev/null
    info "Camera auto-detect enabled in $BOOT_CONFIG"
else
    info "Camera auto-detect already enabled"
fi

# ── 5. Boot-speed hardware trims ────────────
step "Applying boot-speed hardware trims"

ensure_boot_config_line() {
    local key="$1"
    local line="$2"
    if grep -q "^${key}=" "$BOOT_CONFIG" 2>/dev/null; then
        sudo sed -i "s|^${key}=.*|${line}|" "$BOOT_CONFIG"
    else
        echo "$line" | sudo tee -a "$BOOT_CONFIG" > /dev/null
    fi
}

ensure_exact_boot_config_line() {
    local line="$1"
    if ! grep -q "^${line}$" "$BOOT_CONFIG" 2>/dev/null; then
        echo "$line" | sudo tee -a "$BOOT_CONFIG" > /dev/null
    fi
}

ensure_exact_boot_config_line "dtoverlay=disable-bt"
ensure_boot_config_line "dtparam=audio" "dtparam=audio=off"
ensure_boot_config_line "hdmi_blanking" "hdmi_blanking=2"
ensure_boot_config_line "boot_delay" "boot_delay=0"
ensure_boot_config_line "display_auto_detect" "display_auto_detect=0"
ensure_boot_config_line "disable_splash" "disable_splash=1"
ensure_boot_config_line "disable_poe_fan" "disable_poe_fan=1"
ensure_boot_config_line "force_eeprom_read" "force_eeprom_read=0"
ensure_boot_config_line "enable_tvout" "enable_tvout=0"
sudo sed -i 's/^dtparam=audio=on/#dtparam=audio=on # disabled by reFrame fast boot/' "$BOOT_CONFIG"

# Keep kernel messages in the journal while avoiding blocking serial-console
# output and routine console chatter on this completely headless build.
sudo sed -i -E 's/(^|[[:space:]])console=serial0,[^[:space:]]+//g; s/[[:space:]]+/ /g; s/^ //; s/ $//' "$BOOT_CMDLINE"
if ! grep -qw "quiet" "$BOOT_CMDLINE"; then
    sudo sed -i '1 s/$/ quiet/' "$BOOT_CMDLINE"
fi
if ! grep -qw "loglevel=3" "$BOOT_CMDLINE"; then
    sudo sed -i '1 s/$/ loglevel=3/' "$BOOT_CMDLINE"
fi

info "Applied headless Bluetooth, audio, display-probe, console, and boot-delay trims"

# ── 6. User permissions ─────────────────────
step "Setting up user permissions"

sudo usermod -a -G video,i2c,spi,gpio "$USER"
info "Added $USER to hardware groups (video, i2c, spi, gpio)"

# ── 7. Configuration ────────────────────────
step "Setting up configuration"

if [ ! -f "$SCRIPT_DIR/settings.json" ]; then
    cp "$SCRIPT_DIR/settings.example.json" "$SCRIPT_DIR/settings.json"
    info "Created settings.json from example"
else
    info "settings.json already exists, skipping"
fi

# Create photo directories
mkdir -p "$SCRIPT_DIR/photos"
mkdir -p "$SCRIPT_DIR/dithered_photos"
info "Photo directories ready"

# ── 8. Runtime scripts ──────────────────────
step "Setting up runtime scripts"

chmod +x "$SCRIPT_DIR/scripts/enable_hdr.sh"
chmod +x "$SCRIPT_DIR/scripts/reframe-python"
info "Runtime scripts marked executable"

# Install the narrow privileged helper used after dashboard git updates.
sudo install -o root -g root -m 0755 "$SCRIPT_DIR/scripts/reframe-apply-update" /usr/local/sbin/reframe-apply-update
echo "cam ALL=(root) NOPASSWD: /usr/local/sbin/reframe-apply-update" | sudo tee /etc/sudoers.d/reframe-update-helper > /dev/null
sudo chmod 440 /etc/sudoers.d/reframe-update-helper
sudo visudo -cf /etc/sudoers.d/reframe-update-helper > /dev/null
info "Installed software update helper"

# ── 9. Systemd services ─────────────────────
step "Installing systemd services"

# Install the PiSugar RTC synchronization and SD-card WiFi import helpers.
sudo install -o root -g root -m 0755 "$SCRIPT_DIR/scripts/reframe-rtc-sync" /usr/local/sbin/reframe-rtc-sync
sudo install -o root -g root -m 0755 "$SCRIPT_DIR/scripts/reframe-wifi-import" /usr/local/sbin/reframe-wifi-import
sudo cp "$SCRIPT_DIR/systemd/reframe-rtc-restore.service" /etc/systemd/system/reframe-rtc-restore.service
sudo cp "$SCRIPT_DIR/systemd/reframe-rtc-update.service" /etc/systemd/system/reframe-rtc-update.service
sudo cp "$SCRIPT_DIR/systemd/reframe-wifi-import.service" /etc/systemd/system/reframe-wifi-import.service
sudo chmod 644 \
    /etc/systemd/system/reframe-rtc-restore.service \
    /etc/systemd/system/reframe-rtc-update.service \
    /etc/systemd/system/reframe-wifi-import.service
sudo install -d -o root -g root -m 0755 /var/lib/reframe
sudo timedatectl set-timezone UTC
sudo timedatectl set-ntp true

# The TCP command API can change power and RTC settings and has no per-command
# authentication. Keep the web UI reachable on 8421, but bind commands on 8423
# to localhost; reFrame's helper uses PiSugar's root-only Unix socket.
if [ -f /etc/default/pisugar-server ]; then
    sudo sed -i -E 's|--tcp[[:space:]]+0\.0\.0\.0:8423|--tcp 127.0.0.1:8423|' /etc/default/pisugar-server
fi
info "Installed PiSugar RTC boot/network synchronization"

# Install camera service
sudo cp "$SCRIPT_DIR/systemd/reframe.service" /etc/systemd/system/reframe.service
sudo chmod 644 /etc/systemd/system/reframe.service
info "Installed reframe.service"

# Install dashboard service
sudo cp "$SCRIPT_DIR/systemd/reframe-dashboard.service" /etc/systemd/system/reframe-dashboard.service
sudo chmod 644 /etc/systemd/system/reframe-dashboard.service
info "Installed reframe-dashboard.service"

# Install dashboard proxy service (port 80 -> 8000)
sudo cp "$SCRIPT_DIR/systemd/reframe-dashboard-proxy.service" /etc/systemd/system/reframe-dashboard-proxy.service
sudo chmod 644 /etc/systemd/system/reframe-dashboard-proxy.service
info "Installed reframe-dashboard-proxy.service"

# Reload and enable
sudo systemctl daemon-reload
sudo systemctl reenable reframe.service
sudo systemctl enable reframe-dashboard.service
sudo systemctl enable reframe-dashboard-proxy.service
sudo systemctl enable reframe-rtc-restore.service
sudo systemctl enable reframe-rtc-update.service
sudo systemctl enable reframe-wifi-import.service
info "Services enabled (will start on boot)"

# ── 10. Boot-speed service trims ─────────────
step "Disabling nonessential boot services"

# These are reversible trims for a headless camera. WiFi, SSH, mDNS/Avahi,
# PiSugar, and all reFrame services are intentionally left enabled.
sudo systemctl disable --now \
    ModemManager.service \
    triggerhappy.service \
    triggerhappy.socket \
    udisks2.service \
    rsync.service \
    rpi-eeprom-update.service \
    bluetooth.service \
    hciuart.service \
    bluealsa.service \
    networking.service \
    keyboard-setup.service \
    console-setup.service \
    rsyslog.service \
    apt-daily.timer \
    apt-daily-upgrade.timer \
    man-db.timer \
    e2scrub_all.timer \
    fstrim.timer \
    2>/dev/null || true

info "Disabled unused desktop/modem/Bluetooth/update timer services when present"

# ── 11. Quick camera sanity check ───────────
step "Checking camera connection"
CAM_CMD=""
command -v rpicam-hello >/dev/null 2>&1 && CAM_CMD="rpicam-hello"
[ -z "$CAM_CMD" ] && command -v libcamera-hello >/dev/null 2>&1 && CAM_CMD="libcamera-hello"
if [ -n "$CAM_CMD" ]; then
    if "$CAM_CMD" --list-cameras 2>&1 | grep -q "^[[:space:]]*0 :"; then
        info "Camera module detected"
    else
        warn "No camera detected yet."
        warn "If the camera is connected, check that the ribbon cable is fully seated"
        warn "and not flipped upside down at either end (or re-check after reboot)."
    fi
fi

# ── 12. Optional backup WiFi ────────────────
if command -v nmcli >/dev/null 2>&1; then
    echo ""
    echo -ne "  Add a backup WiFi network (e.g. phone hotspot)? [y/N] "
    read -r add_wifi
    if [[ "$add_wifi" =~ ^[Yy]$ ]]; then
        echo -ne "  WiFi name (SSID, 2.4 GHz): "
        read -r wifi_ssid
        echo -ne "  WiFi password: "
        read -rs wifi_pass
        echo ""
        if [ -n "$wifi_ssid" ]; then
            sudo nmcli connection delete "$wifi_ssid" >/dev/null 2>&1 || true
            if [ -n "$wifi_pass" ]; then
                if sudo nmcli connection add type wifi ifname wlan0 con-name "$wifi_ssid" ssid "$wifi_ssid" \
                    wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$wifi_pass" connection.autoconnect yes >/dev/null; then
                    info "Saved backup WiFi network '$wifi_ssid'"
                else
                    warn "Could not save backup WiFi network '$wifi_ssid'"
                fi
            else
                if sudo nmcli connection add type wifi ifname wlan0 con-name "$wifi_ssid" ssid "$wifi_ssid" \
                    connection.autoconnect yes >/dev/null; then
                    info "Saved open backup WiFi network '$wifi_ssid'"
                else
                    warn "Could not save backup WiFi network '$wifi_ssid'"
                fi
            fi
        fi
    fi
fi

# ── Done ────────────────────────────────────
echo ""
echo -e "${GREEN}  ┌─────────────────────────────────────┐${NC}"
echo -e "${GREEN}  │   Installation complete!             │${NC}"
echo -e "${GREEN}  └─────────────────────────────────────┘${NC}"
echo ""
echo "  Next steps:"
echo "    1. Reboot:  sudo reboot"
echo "    2. reFrame will start automatically on boot"
echo "    3. Dashboard: http://$(hostname).local"
echo "       Fallback:  http://$(hostname).local:8000"
echo ""
echo -ne "  Reboot now? [y/N] "
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    sudo reboot
fi
