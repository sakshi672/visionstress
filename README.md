# VisionStress

VisionStress evaluates how object detection models perform under controlled visual degradation. Milestone 1 implements setup, validation, and a baseline benchmark using a pretrained YOLO11 detector and a fixed 128-image subset of COCO val2017. Later milestones will add visual stress, failure analysis, and reporting.

**Status:** Milestone 1 completed and verified on CPU. See [PLAN.md](PLAN.md) for evidence and later milestones.

## Phase 1 scope

- Evaluate a real pretrained, CPU-compatible YOLO detector on a real labeled dataset, with GPU acceleration when available.
- Compare the original images with five stress conditions: low light, Gaussian blur, motion blur, Gaussian noise, and JPEG compression. Each condition has three severity levels, for 16 scenarios including the baseline.
- Measure precision, recall, mAP@0.5, mAP@0.5:0.95, per-class AP where supported, false positives, false negatives, and inference latency using documented evaluation settings.
- Analyze degradation and representative detection failures, then export results, plots, annotated images, and a standalone HTML report.
- Record configuration, seeds, model and dataset hashes, software versions, hardware, predictions, and evaluation settings for reproducibility.

The initial phase does not include a frontend, authentication, or cloud infrastructure. Synthetic corruptions approximate real-world conditions; a small benchmark is not a safety certification.

## Install and run Milestone 1

Use Python 3.11 or newer. In PowerShell, from the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,report]"
.\.venv\Scripts\visionstress.exe setup --config configs/benchmark.yaml
.\.venv\Scripts\visionstress.exe validate --config configs/benchmark.yaml
.\.venv\Scripts\visionstress.exe baseline --config configs/benchmark.yaml
.\.venv\Scripts\python.exe -m pytest
```

Use `python3 -m venv .venv`, `.venv/bin/python`, and `.venv/bin/visionstress` on macOS or Linux. Setup downloads the official YOLO11 medium checkpoint and a [pinned 20 MB mirror of COCO val2017 annotations](https://huggingface.co/datasets/pcuenq/coco2017-instances/blob/main/instances_val2017.json), then downloads the first 128 validation images in sorted image ID order from the official COCO S3 bucket. The full annotation file is discarded after extracting the fixed subset. Downloads may take several minutes. The checkpoint and dataset are stored in ignored `models/` and `data/` directories.

The model URL, SHA-256, image hashes, annotation hash, effective configuration, software versions, hardware, raw predictions, ground truth, and baseline metrics are saved for each run under an ignored, unique `outputs/` directory. `validate` verifies the checkpoint hash before deserialization and checks each image, annotation, box, and class mapping. The configurable model path must point to one of the supported YOLO11 checkpoints (`n`, `s`, or `m`) and its setup manifest.

The verified baseline comparisons on 22 November 2026 used Python 3.13.5, PyTorch 2.15.0+cpu, Ultralytics 8.4.216, and pycocotools 2.0.11. All three used the same images and evaluation settings at 640-pixel input:

| Checkpoint | Config | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Mean inference time (CPU) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| YOLO11n | `configs/benchmark_nano.yaml` | 0.6068 | 0.4522 | 0.7247 | 0.5512 | 102.7 ms/image |
| YOLO11s | `configs/benchmark_small.yaml` | 0.7007 | 0.5316 | 0.7309 | 0.6169 | 171.7 ms/image |
| **YOLO11m (default)** | `configs/benchmark.yaml` | **0.7322** | **0.5741** | 0.7269 | 0.6492 | 430.4 ms/image |

Measured results are in the local, Git-ignored `outputs/20261122T085514Z_78c0cf1a`, `outputs/20261122T093236Z_529dad02`, and `outputs/20261122T093330Z_29cdde77` directories, respectively. The medium checkpoint improves mAP@0.5:0.95 by 0.1219 absolute over nano while taking about 4.2 times as long per image on this CPU. These comparisons use the same 128 images to choose the default model, so treat them as exploratory selection results; confirm generalization on an independent image set before making performance claims.

## Evaluation method

The checkpoints are [Ultralytics YOLO11 detection models](https://docs.ultralytics.com/models/yolo11/) from the [official v8.3.0 asset release](https://github.com/ultralytics/assets/releases/tag/v8.3.0). Verified SHA-256 hashes are `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` (nano), `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` (small), and `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95` (medium). The dataset uses a fixed slice of the [COCO 2017 validation split](https://cocodataset.org/#download). It avoids the COCO128 training images. This is a small development benchmark, so results should not be treated as production estimates.

Inference uses 640-pixel input, confidence 0.001, NMS IoU 0.7, and up to 300 raw detections per image. [COCOeval](https://github.com/cocodataset/cocoapi) computes mAP@0.5 and mAP@0.5:0.95 over IoU 0.50–0.95 in steps of 0.05, with at most 100 detections per image. Per-class AP is reported only for classes with at least five non-crowd examples. Precision, recall, true positives, false positives, and false negatives use COCOeval matching at confidence 0.25 and IoU 0.5, including its crowd and ignore handling. Average model inference time is reported separately in milliseconds per image from Ultralytics timing. These settings are configurable in `configs/benchmark.yaml` and saved in each run.

COCO annotations use [CC BY 4.0](https://cocodataset.org/dataset/termsofuse.htm); individual images retain their owners' rights and are subject to Flickr terms. [Ultralytics YOLO11 models and code](https://docs.ultralytics.com/models/yolo11/) are offered under AGPL-3.0 and Enterprise licenses. Review licensing before commercial distribution.

The remaining CLI commands are planned for later milestones:

```text
visionstress run --config configs/benchmark.yaml
visionstress report --run-id <run_id>
```

The report command is intended to regenerate a report from saved results without rerunning inference. Downloaded datasets, model weights, and generated outputs stay outside version control.

## Development milestones

1. Establish the environment, validate the dataset and class mapping, and complete a real baseline evaluation.
2. Implement and evaluate the 15 deterministic stress scenarios.
3. Analyze and visualize detection failures.
4. Generate reports, complete tests, and verify an end-to-end benchmark.

Baseline evaluation must work before proceeding to stress testing. See [PLAN.md](PLAN.md) for the detailed checklist and acceptance criteria.
