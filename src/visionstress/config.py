"""Config loading and validation."""

from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Configuration does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    required = {"dataset", "model", "inference", "evaluation", "output_root", "seed"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(sorted(missing))}")
    root = path.parent.parent if path.parent.name == "configs" else path.parent
    for section, key in (("dataset", "root"), ("model", "path")):
        value = Path(config[section][key])
        config[section][key] = str((root / value).resolve() if not value.is_absolute() else value)
    output = Path(config["output_root"])
    config["output_root"] = str((root / output).resolve() if not output.is_absolute() else output)
    if not 0 < config["inference"]["confidence"] <= 1:
        raise ValueError("inference.confidence must be in (0, 1]")
    if not 0 < config["evaluation"]["detection_confidence"] <= 1:
        raise ValueError("evaluation.detection_confidence must be in (0, 1]")
    if not 0 < config["evaluation"]["matching_iou"] <= 1:
        raise ValueError("evaluation.matching_iou must be in (0, 1]")
    if config["inference"]["confidence"] > config["evaluation"]["detection_confidence"]:
        raise ValueError("Inference confidence must not exceed evaluation detection confidence")
    if config["dataset"]["image_count"] < 1:
        raise ValueError("dataset.image_count must be positive")
    return config
