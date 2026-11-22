"""Standalone HTML summary generated exclusively from persisted measured results."""

from html import escape
import json
from pathlib import Path


def render_report(runs: list[dict], output: Path) -> Path:
    rows = []
    for run in runs:
        result = run["result"]
        if not result or run["status"] != "completed":
            raise ValueError("Report requires completed measured jobs")
        before, after = result["baseline"], result["stressed"]
        def value(metric):
            return "undefined" if metric is None else f"{metric:.4f}"
        rows.append(f"<tr><td>{escape(run['spec']['stress'])}</td><td>{run['spec']['severity']}</td><td>{result['image_count']}</td><td>{value(before['map50_95'])}</td><td>{value(after['map50_95'])}</td><td>{value(result['degradation']['map50_95']['absolute'])}</td><td>{after['average_inference_ms']:.1f} ms</td></tr>")
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VisionStress demo report</title><style>body{font:16px system-ui;max-width:950px;margin:3rem auto;padding:0 1rem;color:#182232}table{border-collapse:collapse;width:100%}th,td{padding:.8rem;border-bottom:1px solid #ccd5df;text-align:left}th{background:#eaf5f0}p{line-height:1.6}</style></head><body><h1>VisionStress demo report</h1><p>Measured object-detection performance on a small, user-downloaded COCO development subset. Synthetic stress is a diagnostic; this report is not a real-world reliability estimate or safety certification.</p><table><thead><tr><th>Stress</th><th>Severity</th><th>Images</th><th>Baseline mAP50:95</th><th>Stressed mAP50:95</th><th>Absolute change</th><th>Stressed latency</th></tr></thead><tbody>""" + "".join(rows) + "</tbody></table><p>AP uses COCOeval IoU 0.50:0.05:0.95 with up to 100 detections per image. The held-out test images are not used in this quick demo. Full raw outputs and reproducibility metadata are in each job's artifact directory.</p></body></html>"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    return output
