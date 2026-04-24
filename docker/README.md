# loco-mujoco Docker Guide

This document describes how to build and run loco-mujoco containers.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime configured (for GPU support — required by JAX `cuda12` wheel)
- For X11 mode: X11 server running on the host

## Python environment

镜像使用 [uv](https://github.com/astral-sh/uv) 在 `/opt/venv` 安装依赖（构建时 `uv sync --frozen`，与仓库根目录 `pyproject.toml` / `uv.lock` 一致），并额外装 `jax[cuda12]` 以启用 GPU JAX / MJX。`PATH` 已包含 `/opt/venv/bin`，进入容器后直接运行 `python` 即可，无需激活 conda/mamba。

## Build Image

From the project root:

```bash
cd docker
docker-compose -f docker-compose.x11.yaml build
# or
docker-compose -f docker-compose.headless.yaml build
```

With custom image name:

```bash
IMAGE=loco-mujoco:custom docker-compose -f docker-compose.x11.yaml build
```

## Start Container

### X11 mode (GUI display)

For visualization (env rendering, tutorials that open a MuJoCo viewer):

```bash
cd docker
DISPLAY=${DISPLAY} docker-compose -f docker-compose.x11.yaml up -d
```

Or foreground:

```bash
DISPLAY=${DISPLAY} docker-compose -f docker-compose.x11.yaml up
```

### Headless mode

For training, batch evaluation, or other non-GUI use:

```bash
cd docker
docker-compose -f docker-compose.headless.yaml up -d
```

Or foreground:

```bash
docker-compose -f docker-compose.headless.yaml up
```

## Attach to Container

```bash
docker exec -it loco-mujoco_container bash
```

## Stop Container

```bash
cd docker
docker-compose -f docker-compose.x11.yaml down
# or
docker-compose -f docker-compose.headless.yaml down
```

## View Logs

```bash
docker logs loco-mujoco_container
# or follow
docker logs -f loco-mujoco_container
```

## Configuration

### Container name
- Fixed name: `loco-mujoco_container`

### GPU
- Uses all available NVIDIA GPUs by default
- Set `GPU` env var to override (default: `all`)

### Working directory
- Container workdir: `/workspace` (source mounted from repo root)

### Network
- Uses `host` network mode

### Environment variables
- **DISPLAY** (X11 only): X11 display
- **GPU**: GPU selection (default: `all`)
- **HF_HOME**: HuggingFace cache path inside container (default `/root/.cache/huggingface`, mounted to `../.hf_cache` on host)
- **LOCO_AUTO_MYO**: set to `1` to auto-accept the MyoSkeleton license and fetch the model on container start (default `0`)

## Troubleshooting

### X11 permission denied

```bash
xhost +local:docker
```

### GPU not detected

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Container name conflict

```bash
docker stop loco-mujoco_container
docker rm loco-mujoco_container
```

### `uv sync --frozen` fails at build

`uv.lock` is required at the repo root. Regenerate if dependencies changed:

```bash
cd /path/to/loco-mujoco
uv lock
```

### `jax.devices()` only shows CPU

The NVIDIA Docker runtime is not active. Verify with `nvidia-smi` inside the container, and make sure `--gpus all` (compose `deploy.resources.reservations`) is honored by your Docker installation.

## Example Usage

Inside the container (`docker exec -it loco-mujoco_container bash`):

```bash
# The uv-managed venv at /opt/venv is already on PATH — use `python` directly.

# ---------- Smoke test (headless, no dataset, no MyoSkeleton) ----------
# Creates a plain RL env and runs 10 random steps. Does NOT require HuggingFace
# downloads or the license-gated MyoSkeleton model.
python -c "
import numpy as np
from loco_mujoco import RLFactory
env = RLFactory.make('UnitreeH1')
env.reset()
for _ in range(10):
    env.step(np.random.randn(env.info.action_space.shape[0]))
print('smoke test OK')
"

# ---------- GPU / JAX check ----------
python -c "import jax; print(jax.devices())"        # expects [CudaDevice(id=0), ...]
python -c "import loco_mujoco; print(loco_mujoco.__file__)"  # expects /workspace/loco_mujoco/__init__.py

# ---------- Rendering tutorials (need X11 mode + dataset) ----------
# python examples/tutorials/01_creating_mujoco_env.py    # MyoSkeleton + stepinplace1 dataset
# python examples/tutorials/02_creating_mjx_env.py       # GPU MJX + dataset

# ---------- Dataset downloads ----------
loco-mujoco-download-real       # real mocap datasets
loco-mujoco-download-perfect    # perfect / synthetic datasets
loco-mujoco-download            # both of the above

# ---------- MyoSkeleton (license-gated) ----------
# Either set LOCO_AUTO_MYO=1 at compose up, or run manually inside the container:
loco-mujoco-myomodel-init

# ---------- Run training examples ----------
python examples/training_examples/jax_rl/experiment.py
python examples/training_examples/jax_rl_mimic/experiment.py

# ---------- Tests ----------
pytest --ignore=tests/test_task_factories.py
```

## Notes

- X11 mode requires a running X server and `xhost` access for Docker
- First build can take a while (CUDA cuDNN runtime + JAX CUDA wheels)
- `/workspace` is the project root; loco-mujoco is installed editable at entrypoint
- HuggingFace cache persists across container restarts via `../.hf_cache` bind mount on the host
- AMASS / LAFAN1 paths must be configured with `loco-mujoco-set-amass-path` / `loco-mujoco-set-lafan1-path` before use
