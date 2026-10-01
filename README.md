# VOSS: Vectorized Odd Segmented Sieve

**Fastest open-source GPU implementation for computing prime gaps.**

## Overview

VOSS computes prime gaps on NVIDIA GPUs using:
- Wheel-30 bit representation
- CUDA Graphs for batched kernel launches
- Privatized histograms
- Atomic bit-clearing

## Verified Results

| N | Time | Prime Count |
|----|------|-------------|
| 10^9  | 52.8 ms | 50,847,534 |
| 10^10 | 743.9 ms | 455,052,511 |
| 10^11 | 8.24 s | 4,118,054,813 |
| 10^12 | 95.8 s | 37,607,912,018 |
| 10^13 | 694.6 s | 346,065,536,839 |

All values verified against OEIS A006880 and the Prime Gap List Project.

## Repository Structure

```
.
|-- src/                  # CUDA + Python source
|-- scripts/              # Helper scripts
|-- data/                 # Reference data
|   `-- reference/        # Original v7 files
|-- results/              # Output CSVs
|-- bin/                  # Compiled binary (gitignored)
|-- docs/                 # Documentation
`-- figures/              # Charts
```

## Quick Start

```bash
pip install -r requirements.txt
cd src
python3 voss_master.py
```

## Requirements

- NVIDIA GPU (sm_75 / sm_80 / sm_89 / sm_90)
- CUDA Toolkit >= 12.0
- Python >= 3.8

## License

MIT - see LICENSE

## Author

Ataf Hamada - 2026

---
*Generated on 2026-10-01*
