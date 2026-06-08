#!/usr/bin/env bash
# Theta production install — for Linux service deployments (DGX, AI Factory, shared clusters)
#
# Usage:
#   sudo bash install.sh
#
# What this does:
#   1. Creates a 'theta' system user (no login shell)
#   2. Adds it to the 'video' group (NVML GPU access)
#   3. Creates a venv at /opt/theta/venv and installs runtheta into it
#   4. Installs the systemd service unit
#   5. Prints next steps (calibrate, then enable)
#
# This script does NOT start the daemon. Run:
#   sudo -u theta /opt/theta/venv/bin/theta calibrate --gpu 0
#   sudo systemctl enable --now theta
# after it completes.

set -euo pipefail

VENV_DIR="/opt/theta/venv"
CONFIG_DIR="/etc/theta"
SERVICE_FILE="/etc/systemd/system/theta.service"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Preflight ──────────────────────────────────────────────────────────────────

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: This script must be run as root (sudo bash install.sh)" >&2
    exit 1
fi

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ first." >&2
    exit 1
fi

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJ=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MIN=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [[ $PY_MAJ -lt 3 ]] || [[ $PY_MAJ -eq 3 && $PY_MIN -lt 10 ]]; then
    echo "ERROR: Python 3.10+ required (found $PY_VER)" >&2
    exit 1
fi

echo ""
echo "  Theta production installer"
echo "  Python $PY_VER  ·  $(uname -s) $(uname -m)"
echo ""

# ── Step 1: Service user ────────────────────────────────────────────────────────

if id "theta" &>/dev/null; then
    echo "  [✓] User 'theta' already exists"
else
    useradd --system --shell /sbin/nologin --home-dir /var/lib/theta --create-home theta
    echo "  [✓] Created system user 'theta'"
fi

# NVML requires membership in the 'video' group (most distros) or 'nvidia' group.
# Add theta to both; the one that doesn't exist is silently skipped.
for GRP in video nvidia render; do
    if getent group "$GRP" &>/dev/null; then
        usermod -aG "$GRP" theta 2>/dev/null && echo "  [✓] Added theta to '$GRP' group" || true
    fi
done

# ── Step 2: Config directory ────────────────────────────────────────────────────

mkdir -p "$CONFIG_DIR"
chown theta:theta "$CONFIG_DIR"
chmod 750 "$CONFIG_DIR"
echo "  [✓] Config dir $CONFIG_DIR"

# ── Step 3: Virtual environment ─────────────────────────────────────────────────

mkdir -p /opt/theta
python3 -m venv "$VENV_DIR"
chown -R theta:theta /opt/theta
echo "  [✓] Created venv at $VENV_DIR"

# Install runtheta into the venv as the theta user
sudo -u theta "$VENV_DIR/bin/pip" install --quiet --upgrade pip
sudo -u theta "$VENV_DIR/bin/pip" install --quiet runtheta
echo "  [✓] Installed runtheta → $VENV_DIR/bin/theta"

THETA_BIN="$VENV_DIR/bin/theta"

# ── Step 4: Systemd service ─────────────────────────────────────────────────────

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Theta GPU Thermal-Power Forensics Agent
Documentation=https://github.com/Asomisetty27/theta
After=network.target nvidia-dcgm.service
Wants=nvidia-dcgm.service

[Service]
Type=simple
User=theta
Group=theta
ExecStart=$THETA_BIN monitor --quiet
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=theta

# Config and calibration — shared paths so the service user finds them
Environment=THETA_CONFIG_DIR=$CONFIG_DIR

# Resource limits
LimitNOFILE=65536
MemoryMax=512M

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
echo "  [✓] Installed systemd unit → $SERVICE_FILE"

# ── Step 5: Verify NVML access ─────────────────────────────────────────────────

echo ""
echo "  Checking NVML access for 'theta' user…"
if sudo -u theta "$VENV_DIR/bin/python3" -c "
import pynvml
try:
    pynvml.nvmlInit()
    n = pynvml.nvmlDeviceGetCount()
    names = [pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(i)) for i in range(n)]
    pynvml.nvmlShutdown()
    print(f'  [✓] NVML OK — {n} GPU(s): {names}')
except pynvml.NVMLError as e:
    print(f'  [!] NVML error: {e}')
    print('      If this is a permission error, log out and back in, or reboot.')
    exit(1)
" 2>&1; then
    NVML_OK=1
else
    NVML_OK=0
fi

# ── Summary ─────────────────────────────────────────────────────────────────────

echo ""
echo "  ─────────────────────────────────────────────"
echo "  Installation complete."
echo ""
echo "  REQUIRED before starting the daemon:"
echo ""
echo "  1. Run calibration as the theta service user:"
echo "     (DGX / AI Factory — always-busy node):"
echo "       sudo -u theta $THETA_BIN calibrate --gpu 0 \\"
echo "         --ambient <coolant_inlet_temp_c> \\"
echo "         --calibration-file $CONFIG_DIR/calibration.json"
echo ""
echo "     (node has idle windows available):"
echo "       sudo -u theta $THETA_BIN calibrate --gpu 0 \\"
echo "         --calibration-file $CONFIG_DIR/calibration.json"
echo ""
echo "     Repeat for each GPU index (0, 1, 2, …)."
echo ""
echo "  2. Run setup wizard to configure alerting:"
echo "       sudo -u theta $THETA_BIN setup"
echo ""
echo "  3. Enable and start the service:"
echo "       sudo systemctl enable --now theta"
echo "       sudo journalctl -u theta -f"
echo ""
echo "  Prometheus metrics: http://localhost:9101/metrics"
echo "  Health API:         http://localhost:9102/api/v1/health"
echo ""
echo "  Ports to open in firewall (if scraping externally):"
echo "    9101/tcp — Prometheus metrics"
echo "    9102/tcp — Health API (bearer-token auth required)"
echo "  ─────────────────────────────────────────────"
echo ""
