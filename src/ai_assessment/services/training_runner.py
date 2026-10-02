"""Child process entry point for one local Ultralytics train/val run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _json_value(value.tolist())
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _per_class(metrics: Any) -> list[dict[str, Any]]:
    box = getattr(metrics, "box", None)
    if box is None:
        return []
    names = getattr(metrics, "names", {}) or {}
    ap = getattr(box, "ap", None)
    ap50 = getattr(box, "ap50", None)
    precision = getattr(box, "p", None)
    recall = getattr(box, "r", None)
    rows: list[dict[str, Any]] = []
    count = max((len(value) for value in (ap, ap50, precision, recall) if value is not None), default=0)
    for index in range(count):
        def entry(value: Any) -> Any:
            try:
                return _json_value(value[index])
            except (IndexError, TypeError):
                return None
        rows.append({"class_id": index, "class_name": str(names.get(index, index)), "map50": entry(ap50), "map50_95": entry(ap), "precision": entry(precision), "recall": entry(recall)})
    return rows


def _prepare_dataset_yaml(source_yaml: Path, result_dir: Path) -> Path:
    """Build a run-owned YAML with an absolute dataset root.

    Older M3 exports contain path: ., which Ultralytics resolves relative
    to its configured datasets directory. Leave the exported YAML untouched.
    """
    source = source_yaml.expanduser().resolve(strict=True)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {source}")
    data["path"] = str(source.parent)
    result_dir.mkdir(parents=True, exist_ok=True)
    resolved_yaml = result_dir / "resolved_data.yaml"
    resolved_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return resolved_yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args(argv)
    from ultralytics import YOLO

    resolved_data = _prepare_dataset_yaml(Path(args.data), Path(args.result_json).parent)
    print(f"Resolved dataset root: {Path(args.data).resolve().parent}", flush=True)
    model = YOLO(args.weights)
    train_kwargs = {
        "data": str(resolved_data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "workers": args.workers,
        "patience": args.patience,
        "project": args.project,
        "name": args.name,
        "exist_ok": False,
        "verbose": True,
        "amp": args.amp,
    }
    if args.device:
        train_kwargs["device"] = args.device
    print(f"Starting train with local weights: {args.weights}", flush=True)
    print(f"Training precision: {'AMP' if args.amp else 'FP32 (AMP check disabled)'}", flush=True)
    model.train(**train_kwargs)
    print("Training complete; validating on split=val", flush=True)
    val_kwargs = {"data": str(resolved_data), "split": "val", "project": args.project, "name": f"{args.name}_val", "exist_ok": False, "plots": True, "verbose": True}
    if args.device:
        val_kwargs["device"] = args.device
    validation = model.val(**val_kwargs)
    payload = {
        "metrics": _json_value(getattr(validation, "results_dict", {}) or {}),
        "per_class": _per_class(validation),
        "confusion_matrix": str(Path(args.project) / f"{args.name}_val" / "confusion_matrix.png"),
    }
    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote metrics: {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
