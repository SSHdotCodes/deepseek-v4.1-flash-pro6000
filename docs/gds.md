# GPUDirect Storage

About a third of the routed experts are not in VRAM or RAM and are read from NVMe when a token routes to them. Without GDS, each read lands in pinned host memory and is then copied to the GPU, so the bytes cross host DRAM twice, competing with the RAM tier's own uploads for about 48 GB/s of DRAM bandwidth. With GDS P2P, cuFile has the NVMe drive write straight into the expert's VRAM slot over PCIe.

On the reference machine, with uploads from RAM running at the same time, P2P cut an NVMe expert read from 2.76 to 1.69 ms and raised the concurrent upload rate from 34 to 45 GB/s. End to end, on a fresh clone of this repo, decode measured 31.61 tok/s with P2P and 30.79 without; all 12 paired prompts were faster with P2P (+0.4% to +5.0%, median +2.8%), with identical tokens ([results](../benchmarks/clean-clone-20261018/)).

## Enabling P2P

Requirements: the checkpoint on a local NVMe drive (not RAID or NVMe-oF), the CUDA 13.4 toolkit's GDS libraries (`libcufile.so.0`). Measured with ext4.

```bash
sudo bash gds/enable-p2p.sh     # writes /etc/modprobe.d/ds41-gds-p2p.conf, rebuilds the initramfs, reboots
```

The file sets two options and nothing else:

- `nvidia NVreg_RegistryDwords="RMForceStaticBar1=1;RmForceDisableIomapWC=1;"`: map all of VRAM in BAR1 statically so the drive can write into it, without write-combined mappings;
- `nvme_core multipath=N`: plain NVMe block devices, which cuFile's P2PDMA path requires.

`sudo bash gds/revert-p2p.sh` removes the file and reboots. With `"gds": "auto"` (the default), `scripts/run.py` turns GDS on when it finds `RMForceStaticBar1=1` in the loaded driver's options.

## ext4 and `data=ordered`

cuFile only uses P2P on ext4 when the mount table shows `data=ordered`. That is ext4's default mode, but Linux prints `data=` only when it was given explicitly at mount time, so a default root filesystem looks unsupported and every read silently falls back to cuFile's POSIX path. Two fixes:

- **Shim (automatic).** `build/dist/mountinfo_shim.so` is preloaded into `ds41d` when the checkpoint's filesystem is ext4 without a visible `data=` option. It adds `,data=ordered` to ext4 lines of the mount tables the process reads, and changes nothing else. An ext4 mount without a `data=` option is in ordered mode; `/proc/fs/ext4/<device>/options` confirms it.
- **Mount option.** Mount the filesystem with `data=ordered` explicitly (for the root filesystem, add `rootflags=data=ordered` to the kernel command line). The shim is then not used.

## Checking it

While the service runs:

```bash
/usr/local/cuda-13.4/gds/tools/gds_stats -p "$(pgrep -f 'ds41d serve')" -l 3 | grep p2pdma
```

`p2pdma=N posix=0` means reads go GPU-direct. `p2pdma=0 posix=N` means the fallback is in use: check the mount and the driver options. The engine logs cuFile errors to `$XDG_RUNTIME_DIR/<unit>/cufile.log`.

`"gds": false` in `config.local.json` turns GDS off. NVMe reads then use `O_DIRECT` into pinned staging buffers; tokens do not change.
