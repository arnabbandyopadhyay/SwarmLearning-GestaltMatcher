# Swarm Learning — GestaltMatcher (PyTorch)

Distributed training of a **GestaltMatcher**-style face-embedding classifier on the
**GMDB (GestaltMatcherDB)** dataset using **HPE Swarm Learning**. Each peer trains
locally on its own private data and periodically merges model weights with the other
peers through the Swarm sync mechanism — the raw data never leaves the node.

---

## Repository structure

```
Swarm_learning/
├── swarm.ipynb                 # SWCI orchestration notebook (drives the whole workflow)
├── swci/
│   └── taskdefs/               # SWCI task definitions (YAML)
│       ├── pull_GM_user_image.yaml        # PULL_IMAGE  – pulls the prebuilt "gmdocker" image
│       ├── build_pyt_user_image_gm.yaml   # MAKE_USER_CONTAINER – builds PyTorch + Swarm client image
│       ├── swarm_gm_task_001.yaml         # RUN_SWARM   – 1 peer  (prereq: pull image)
│       └── swarm_gm_task_002.yaml         # RUN_SWARM   – 2 peers (prereq: build image)
└── model/                      # Training code mounted into the Swarm workspace
    ├── train_gm_arc.py         # Main training entry point (Swarm-enabled)
    ├── pyt_ab_gm.py            # PyTorch SwarmCallback (subclasses SwarmCallbackBase)
    ├── lookup_table_gmdb_v1.1.0.txt      # index → disorder ID mapping
    ├── saved_models/
    │   ├── glint360k_r50.onnx   # Pretrained ArcFace backbone (ResNet-50)
    │   ├── glint360k_r100.onnx  # Pretrained ArcFace backbone (ResNet-100)
    └── lib/
        ├── models/             # MyArcFace wrapper, RetinaFace detector, network defs
        ├── datasets/           # GestaltMatcherDataset (GMDB), mapping, transforms
        ├── evaluation.py       # Evaluation / verification helpers
        ├── encode.py           # Face encoding utilities
        ├── face_alignment.py   # Face alignment
        └── utils/              # Prior boxes, NMS, box utilities
```

## Components

| Component | Purpose |
|---|---|
| `swarm.ipynb` | Connects to the SWCI server, creates a context, builds/pulls the user container, registers/finalizes tasks, assigns them to the task runner with a peer count, and polls status until done. |
| `swci/taskdefs/*.yaml` | Task pipeline: **pull/build image → run swarm training**. `swarm_gm_task_001` runs with `MIN_PEERS=2`, `swarm_gm_task_002` with `MIN_PEERS=2`. |
| `model/train_gm_arc.py` | Loads the pretrained ONNX backbone, attaches a 512-d feature layer + classifier, trains with weighted cross-entropy, and syncs weights via `SwarmCallback` every 32 batches (`swSyncInterval`). |
| `model/pyt_ab_gm.py` | PyTorch-specific `SwarmCallback` — saves/loads state dicts during sync rounds, computes local loss/metrics for the SLM UI (`CrossEntropyLoss` / `F1Score`). |

## Prerequisites

- HPE **Swarm Learning** installation ([https://github.com/HewlettPackard/swarm-learning])(https://github.com/HewlettPackard/swarm-learning) with a running SWCI server and at least one
  task runner/peer node per participant.
- Docker (or equivalent) on the nodes that build/pull the user container.
- The Swarm Learning PyTorch client wheel
  (`swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl`) — copied into the
  build context by `build_pyt_user_image_gm.yaml`.
- Python 3 with PyTorch, `onnx2torch`, `torchmetrics`, `albumentations`,
  `scikit-learn`, `tensorboard`, `opencv-python`, etc. (all installed by the
  container build task).

## Data

The GMDB dataset is expected under the directory given by `--data_dir`
(default `data/`), mounted into the container at `/tmp/workspace/model/data`:

```
data/GestaltMatcherDB/v1.1.0/
├── gmdb_align/                          # aligned face images (…_aligned.*)
└── gmdb_metadata/
    ├── gmdb_train_images_v1.1.0.csv     # training split
    └── gmdb_val_images_v1.1.0.csv       # validation split
```

## Configuration

Edit the placeholders in the task definitions before running:

- `swci/taskdefs/pull_GM_user_image.yaml` — registry location, image name, credentials.
- `swci/taskdefs/swarm_gm_task_00*.yaml` — `Src: location of data` → your actual data path,
  plus `MAX_EPOCHS` and `MIN_PEERS` env vars.
- `swarm.ipynb` — SWCI server name/IP, ports (`30304`/`30306`), and TLS cert paths
  under `/tmp/swarm/…`.

Environment variables consumed by `train_gm_arc.py`:

| Variable | Default | Meaning |
|---|---|---|
| `MAX_EPOCHS` | 2 | Total local epochs (also passed to `SwarmCallback`) |
| `MIN_PEERS` | 1 | Minimum peers required per sync round |
| `swarm_ml_hostname` | — | Used to name the per-host log file `<host>_demo.txt` |

## Usage

### 1. Orchestrate via the notebook (swarm mode)

Run `swarm.ipynb` top-to-bottom inside the SWCI client environment:

1. **Connect** — instantiate `swci.Swci(...)` with server, port, and certs; verify
   with `pwd()` / `ls()`.
2. **Context** — `createContext(...)` + `switchContext(...)`.
3. **Container** — pick `swarm_task_current = "build_pyt_user_image_gm"` (or
   `"pull_GM_user_image"`), then `createTaskFrom(...)` → `finalizeTask(...)` →
   `assignTask(..., peers=N)` and poll `isTaskDone(...)` until finished.
4. **Train** — set `swarm_task_current = "swarm_gm_task_001"` (or `_002`), repeat
   create/finalize/assign, and poll until the task completes.
5. **Reset** — `resetTrainingContract(...)` + `resetTaskRunner(...)` between runs.

### 2. Run locally (single node, no Swarm)

```bash
cd model
python train_gm_arc.py --paper_model a --session 1
# or full config:
python train_gm_arc.py --paper_model b --epochs 50 --session 2 --dataset gmdb \
    --in_channels 3 --img_size 112 --use_tensorboard --local --data_dir ../data
```

- `--paper_model a` → `glint360k_r50`, batch size 64, lr 5e-4
- `--paper_model b` → `glint360k_r100`, batch size 128, lr 1e-3

Checkpoints are written to `model/saved_models/` as
`s<session>_<model>_512d_<dataset>…_e<epoch>.pt`.

## Notes

- Swarm sync happens every **32 batches** (`swSyncInterval` in `train_gm_arc.py`).
- The base backbone is frozen by default; the 512-d feature layer and classifier
  head are trained (use `--unfreeze` to unfreeze the base).
- Per-run stdout is redirected to `<swarm_ml_hostname>_demo.txt`.
- Default contract name: `defaultbb.cqdb.sml.hpe`.
