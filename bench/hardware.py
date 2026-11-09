"""Write HARDWARE.md: the exact GPU, driver, CUDA, and library versions in use."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

import torch


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "n/a"


def main() -> None:
    import transformers
    import safetensors

    lines = ["# Hardware", "", "Every number in `bench/results/` was produced on this configuration.", ""]
    lines += ["| | |", "|---|---|"]
    lines += [f"| platform | {platform.platform()} |", f"| python | {platform.python_version()} |",
              f"| torch | {torch.__version__} |", f"| transformers | {transformers.__version__} |",
              f"| safetensors | {safetensors.__version__} |"]
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        lines += [f"| gpu | {p.name} ({p.total_memory / 2**30:.1f} GiB, sm_{p.major}{p.minor}) |",
                  f"| cuda (torch) | {torch.version.cuda} |",
                  f"| cudnn | {torch.backends.cudnn.version()} |",
                  f"| driver | {sh(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'])} |"]
    elif torch.backends.mps.is_available():
        lines += [f"| accelerator | Apple MPS ({sh(['sysctl', '-n', 'machdep.cpu.brand_string'])}) |"]
    else:
        lines += ["| accelerator | none (CPU) |"]
    lines += [f"| git | {sh(['git', 'rev-parse', '--short', 'HEAD'])} |", ""]
    Path("HARDWARE.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
