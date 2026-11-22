# VisionStress Phase 1 completion checklist

Update checkboxes only when the work is implemented and verified. Record blockers and verification evidence below as work progresses. Baseline evaluation is a prerequisite for milestone 2.

## Milestone 1: Environment and baseline

- [ ] Establish a Python 3.11+ virtual environment, project structure, dependency compatibility, and version-control exclusions for datasets, weights, and outputs.
- [ ] Select a trusted, CPU-compatible pretrained YOLO checkpoint; document its source, version, SHA-256, and licensing, including commercial-use considerations.
- [ ] Select a real, publicly available labeled detection dataset of approximately 100–300 images; document provenance and licensing.
- [ ] Implement setup and configuration for the model and dataset, with useful errors if downloads are unavailable.
- [ ] Validate image/annotation pairing, bounding boxes, class IDs, and model-to-dataset class mappings.
- [ ] Implement configurable inference and an established detection evaluator; document IoU thresholds, confidence handling, class mappings, and matching settings.
- [ ] Run real baseline inference and calculate precision, recall, mAP@0.5, mAP@0.5:0.95, supported per-class AP, false-positive/false-negative counts, and separate inference latency.
- [ ] Save baseline predictions, ground-truth references, metrics, and reproducibility metadata; verify the baseline before continuing.

## Milestone 2: Visual stress testing

- [ ] Implement low light, Gaussian blur, motion blur, Gaussian noise, and JPEG compression, each with three documented numeric severity levels.
- [ ] Verify seeded determinism, preserved image dimensions and geometry, and consistent color spaces; save exact parameters for every transformed image.
- [ ] Evaluate the same images with the same model and settings across all 15 stressed scenarios, saving raw predictions and metrics.
- [ ] Calculate absolute and relative changes, handle zero or undefined baselines, compare severity and class impacts, and rank conditions by measured impact.

## Milestone 3: Failure analysis

- [ ] Identify false positives, false negatives, localization errors, and detections that become incorrect under stress using documented matching rules.
- [ ] Rank failures using lost true positives, newly introduced false negatives, and localization deterioration.
- [ ] Preserve original and transformed images, ground truth, baseline and stressed predictions, confidence scores, scenario details, and evaluation settings.
- [ ] Generate annotated before-and-after comparisons and a gallery of up to 20 informative failures.

## Milestone 4: Reporting and verification

- [ ] Generate readable baseline, severity, degradation, per-class, and failure-gallery visualizations from measured results.
- [ ] Produce a standalone HTML report with executive summary, methodology, results, failure analysis, and limitations; export CSV and JSON.
- [ ] Implement report regeneration from saved results without rerunning inference.
- [ ] Save each run in a unique directory with configuration, seeds, model and dataset hashes and manifest, versions, hardware, inference settings, transform parameters, predictions, results, plots, and report.
- [ ] Test transformations, shape preservation, severity validation, annotations, class mapping, known evaluator results, degradation calculations, report generation, and CLI configuration with appropriate fixtures.
- [ ] Run unit tests and at least one actual end-to-end benchmark when model and dataset are available; resolve failures.
- [ ] Add exact, verified installation and execution commands to the README.

## Completion criteria

- [ ] A real pretrained YOLO model and real labeled dataset have been evaluated across baseline and all 15 stress scenarios.
- [ ] Metrics and reproducibility artifacts have been saved, and a report has been generated from measured results.
- [ ] At least one complete benchmark and the unit tests have passed.
- [ ] Remaining limitations, licensing considerations, and any unverified work are documented without fabricated results.

## Progress and blockers

- Planning documents created; implementation and end-to-end verification have not started.
- No model, dataset, benchmark results, or dependency compatibility have been verified yet.
