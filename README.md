# VisionStress

VisionStress is a planned toolkit for evaluating how object detection models perform under controlled visual degradation. Phase 1 will run a pretrained YOLO model against a small, labeled dataset and produce reproducible metrics, failure examples, and a local HTML report.

**Status:** Planning only. The pipeline, dependencies, dataset setup, and CLI have not yet been implemented or verified. Track implementation and blockers in [PLAN.md](PLAN.md).

## Phase 1 scope

- Evaluate a real pretrained, CPU-compatible YOLO detector on a real labeled dataset, with GPU acceleration when available.
- Compare the original images with five stress conditions: low light, Gaussian blur, motion blur, Gaussian noise, and JPEG compression. Each condition has three severity levels, for 16 scenarios including the baseline.
- Measure precision, recall, mAP@0.5, mAP@0.5:0.95, per-class AP where supported, false positives, false negatives, and inference latency using documented evaluation settings.
- Analyze degradation and representative detection failures, then export results, plots, annotated images, and a standalone HTML report.
- Record configuration, seeds, model and dataset hashes, software versions, hardware, predictions, and evaluation settings for reproducibility.

The initial phase does not include a frontend, authentication, or cloud infrastructure. Synthetic corruptions approximate real-world conditions; a small benchmark is not a safety certification.

## Planned workflow

Python 3.11 or newer and a virtual environment are planned. Setup will obtain or configure a trusted pretrained checkpoint and a compatible, publicly available labeled dataset. Model and dataset provenance and licenses must be verified before use; Ultralytics licensing must be reviewed before commercial distribution.

The intended CLI is shown below as a design target. These commands are **not available yet**; exact installation and execution instructions will be added once implemented and tested.

```text
visionstress setup
visionstress validate --config configs/benchmark.yaml
visionstress run --config configs/benchmark.yaml
visionstress report --run-id <run_id>
```

The report command is intended to regenerate a report from saved results without rerunning inference. Downloaded datasets, model weights, and generated outputs should remain outside version control.

## Development milestones

1. Establish the environment, validate the dataset and class mapping, and complete a real baseline evaluation.
2. Implement and evaluate the 15 deterministic stress scenarios.
3. Analyze and visualize detection failures.
4. Generate reports, complete tests, and verify an end-to-end benchmark.

Baseline evaluation must work before proceeding to stress testing. See [PLAN.md](PLAN.md) for the detailed checklist and acceptance criteria.
