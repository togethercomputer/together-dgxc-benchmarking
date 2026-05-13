# Work Log

---

## 2026-05-01

### dgxc-benchmarking Installation on slinky B200 Cluster

**Goal**: Install NVIDIA dgxc-benchmarking and run Llama3.1 training benchmarks on the slinky cluster (51 × B200 nodes).

**Completed**:
- Cloned repo to `/data/home/johnson/dgxc-benchmarking`
- Set install root: `LLMB_INSTALL=/data/home/johnson/llmb`
- Downloaded all 4 NeMo container images (127 GB total) to `/data/home/johnson/llmb/images/`
  - `nvidia+nemo+25.07.01.sqsh` (27 GB)
  - `nvidia+nemo+25.09.00.sqsh` (30 GB)
  - `nvidia+nemo+26.02.00.sqsh` (36 GB) ← B200 target
  - `nvidia+nemo+26.02.01.sqsh` (36 GB)
- Installed all 7 workloads: pretrain_llama3.1, pretrain_deepseek-v3, pretrain_gpt_oss, pretrain_grok1, pretrain_nemotron-h, pretrain_nemotron4-340b, pretrain_qwen3
- Configured enroot paths to WekaFS for persistence (added to `~/.bashrc`)

**Issues resolved**:
| Issue | Fix |
|-------|-----|
| `git lfs` not found | Downloaded static binary to `~/.local/bin/git-lfs`, cleared bash hash |
| System Python 3.10 rejected by installer | Added uv-managed Python 3.12 to PATH in `~/.bashrc` |
| `enroot import` exit code 17 (whiteout failure) | Removed leftover 0-byte `.sqsh` file; fixed `ENROOT_SQUASH_OPTIONS` |
| `enroot import` exit code 1 (invalid option) | Removed `-no-devs` (not valid for `enroot-mksquashovlfs`) |
| `enroot import` timeout (35 min) | Patched `llmb_install/downloads/image.py`: `"35"` → `"120"` minutes |
| `python3-venv` missing | `sudo apt-get install python3.10-venv` |
| pip 22.0.2 AssertionError in resolvelib | Patched `venv_manager.py` to upgrade pip after venv creation |

**Next**: Run first benchmark — `llmb-run submit -w pretrain_llama3.1 -s 8b --dtype fp8 --scale 8` — and record TFLOPS/GPU result.

---
