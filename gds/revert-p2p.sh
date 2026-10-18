#!/bin/bash
# Undo enable-p2p.sh: remove the modprobe file, rebuild the initramfs, reboot.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
rm -f /etc/modprobe.d/ds41-gds-p2p.conf
update-initramfs -u -k all
echo "reverted. Rebooting in 10 s."; sleep 10; systemctl reboot
