#!/bin/bash
# DS41 GPUDirect Storage P2P (NVMe -> VRAM without the host DRAM bounce).
# Run on the server:  sudo bash gds/enable-p2p.sh
# Adds one new modprobe file (nothing else is edited), rebuilds the initramfs, reboots.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
CONF=/etc/modprobe.d/ds41-gds-p2p.conf
cat > "$CONF" <<"CONF_EOF"
# DS41 GPUDirect Storage P2P. Revert: sudo bash gds/revert-p2p.sh
options nvidia NVreg_RegistryDwords="RMForceStaticBar1=1;RmForceDisableIomapWC=1;"
options nvme_core multipath=N
CONF_EOF
echo "wrote $CONF:"; cat "$CONF"
update-initramfs -u -k all
echo "initramfs updated. Rebooting in 10 s (Ctrl-C to postpone; the change applies at next boot)."
sleep 10
systemctl reboot
