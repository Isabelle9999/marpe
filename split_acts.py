# one-off conversion: split acts.pt into per-layer files using mmap=True
# do not load the full 72gb tensor into ram

import sys
from pathlib import Path

import torch

ART = Path(__file__).parent / "artifacts" / "probes"
N_LAYERS = 28


def main():
    src = ART / "acts.pt"
    print(f"opening {src} with mmap=True")
    try:
        acts = torch.load(src, mmap=True, weights_only=True)
    except Exception as e:
        print(f"mmap load failed: {e}")
        print("do not attempt to load 72gb without mmap; aborting")
        sys.exit(1)

    for layer_idx in range(N_LAYERS):
        dst = ART / f"acts_layer_{layer_idx:02d}.pt"
        print(f"layer {layer_idx:2d} -> {dst.name}", end=" ... ", flush=True)
        torch.save(acts[layer_idx], dst)
        print("saved")

    resp = input("\ndelete acts.pt? [y/N] ").strip().lower()
    if resp == "y":
        src.unlink()
        print("deleted acts.pt")
    else:
        print("keeping acts.pt")


if __name__ == "__main__":
    main()
