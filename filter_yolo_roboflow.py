#!/usr/bin/env python3
import argparse
import ast
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]


@dataclass
class SplitPaths:
    split_key: str
    split_name: str
    images_dir: Path
    labels_dir: Path


@dataclass
class RunSpec:
    run_key: str
    run_dir: Path
    categories: Set[int]
    category_names: List[str]


@dataclass
class RunResult:
    run_key: str
    run_dir: Path
    categories: List[str]
    copy_stats: Counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Roboflow YOLO dataset by category and copy selected results"
    )
    parser.add_argument("--data-yaml", type=Path, required=True, help="Path to Roboflow data.yaml")

    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--target-names",
        type=str,
        help="Target category names, comma-separated, e.g. Excavator,Gloves",
    )
    target_group.add_argument(
        "--target-ids",
        type=str,
        help="Target category ids, comma-separated, e.g. 0,1",
    )

    parser.add_argument("--output-root", type=Path, required=True, help="Output root directory")
    parser.add_argument(
        "--merge",
        choices=["true", "false"],
        default="false",
        help="Merge multi-category output to one directory or keep separate",
    )

    parser.add_argument("--train-images", type=Path, default=None, help="Override train images directory")
    parser.add_argument("--val-images", type=Path, default=None, help="Override valid images directory")
    parser.add_argument("--test-images", type=Path, default=None, help="Override test images directory")

    parser.add_argument("--train-labels", type=Path, default=None, help="Override train labels directory")
    parser.add_argument("--val-labels", type=Path, default=None, help="Override valid labels directory")
    parser.add_argument("--test-labels", type=Path, default=None, help="Override test labels directory")

    parser.add_argument("--dry-run", action="store_true", help="Preview only, do not copy/write")
    parser.add_argument("--debug", action="store_true", help="Print detailed path and processing info")
    parser.add_argument(
        "--print-output-dir",
        action="store_true",
        help="Print OUTPUT_DIR:<path> lines at the end (used by pipeline wrappers).",
    )
    return parser.parse_args()


def _alert(message: str) -> None:
    print("!" * 90)
    print(f"[ALERT] {message}")
    print("!" * 90)


def _warn(message: str) -> None:
    print("*" * 90)
    print(f"[WARNING] {message}")
    print("*" * 90)


def _sanitize_name(name: str) -> str:
    cleaned = name.strip().replace("/", "_").replace("\\", "_").replace(" ", "-")
    return cleaned if cleaned else "unknown"


def _short_dataset_name(data_yaml: Path) -> str:
    base = data_yaml.parent.name
    if len(base) <= 20:
        return base
    return f"{base[:10]}.{base[-10:]}"


def _parse_simple_yaml(path: Path) -> Dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    data: Dict[str, Any] = {}

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.split("#", 1)[0].rstrip()
        i += 1
        if not line.strip():
            continue

        if ":" not in line:
            continue

        if line.lstrip() != line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key != "names":
            if value.startswith("[") or value.startswith("{"):
                data[key] = ast.literal_eval(value)
            elif value.isdigit():
                data[key] = int(value)
            elif value:
                data[key] = value.strip("'\"")
            else:
                data[key] = None
            continue

        if value:
            if value.startswith("[") or value.startswith("{"):
                data[key] = ast.literal_eval(value)
            else:
                data[key] = value.strip("'\"")
            continue

        names_list: List[str] = []
        names_map: Dict[int, str] = {}
        while i < len(lines):
            next_raw = lines[i]
            next_line = next_raw.split("#", 1)[0].rstrip()
            if not next_line.strip():
                i += 1
                continue
            if next_line.lstrip() == next_line:
                break

            stripped = next_line.strip()
            if stripped.startswith("-"):
                item = stripped[1:].strip().strip("'\"")
                names_list.append(item)
            elif ":" in stripped:
                idx_raw, name_raw = stripped.split(":", 1)
                idx_raw = idx_raw.strip()
                name = name_raw.strip().strip("'\"")
                if idx_raw.isdigit():
                    names_map[int(idx_raw)] = name
            i += 1

        if names_map:
            data[key] = names_map
        else:
            data[key] = names_list

    return data


def load_data_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"data.yaml not found: {path}")

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("data.yaml content is not a mapping")
        return data
    except ModuleNotFoundError:
        data = _parse_simple_yaml(path)
        if not data:
            raise ValueError("Failed to parse data.yaml without PyYAML")
        return data


def parse_names(names_obj: Any, nc: Optional[int]) -> List[str]:
    if isinstance(names_obj, list):
        names = [str(x) for x in names_obj]
    elif isinstance(names_obj, dict):
        numeric_keys = [k for k in names_obj.keys() if str(k).isdigit()]
        if not numeric_keys:
            raise ValueError("Invalid names mapping in data.yaml")
        max_idx = max(int(k) for k in numeric_keys)
        names = [""] * (max_idx + 1)
        for k, v in names_obj.items():
            if str(k).isdigit():
                names[int(k)] = str(v)
    else:
        raise ValueError("data.yaml names must be list or mapping")

    if nc is not None and isinstance(nc, int) and nc > 0 and len(names) < nc:
        names.extend([f"class_{i}" for i in range(len(names), nc)])

    return names


def resolve_targets(target_names: Optional[str], target_ids: Optional[str], names: List[str]) -> Tuple[List[int], List[str]]:
    if target_names:
        raw_names = [x.strip() for x in target_names.split(",") if x.strip()]
        if not raw_names:
            raise ValueError("target names are empty")
        resolved_ids: List[int] = []
        for item in raw_names:
            if item not in names:
                raise ValueError(f"target name not found in data.yaml names: {item}")
            resolved_ids.append(names.index(item))
        unique_ids = sorted(set(resolved_ids))
        return unique_ids, [names[i] if i < len(names) else f"class_{i}" for i in unique_ids]

    assert target_ids is not None
    raw_ids = [x.strip() for x in target_ids.split(",") if x.strip()]
    if not raw_ids:
        raise ValueError("target ids are empty")

    resolved: List[int] = []
    for item in raw_ids:
        if not item.isdigit():
            raise ValueError(f"target id is not integer: {item}")
        resolved.append(int(item))

    unique_ids = sorted(set(resolved))
    target_names_resolved = [names[i] if 0 <= i < len(names) and names[i] else f"class_{i}" for i in unique_ids]
    return unique_ids, target_names_resolved


def resolve_roboflow_relative(data_yaml_dir: Path, value: str) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute():
        return p

    txt = value.strip()
    candidates: List[Path] = []

    if txt.startswith("../"):
        candidates.append((data_yaml_dir / txt[3:]).resolve())

    candidates.append((data_yaml_dir / txt).resolve())

    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _derive_labels_from_images(images_dir: Path) -> Optional[Path]:
    parts = list(images_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        replaced = parts[:]
        replaced[idx] = "labels"
        return Path(*replaced)

    text = str(images_dir)
    if "/images" in text:
        return Path(text.replace("/images", "/labels", 1))
    return None


def resolve_split_paths(args: argparse.Namespace, data_yaml: Dict[str, Any], data_yaml_path: Path) -> Tuple[List[SplitPaths], List[str], List[str]]:
    """
    解析各 split（train/val/test）的 images/labels 目录。

    返回值：(有效 split 列表, 致命错误列表, 非致命警告列表)
    - 致命错误：用户手动指定（--val-images 等）的路径不存在，或完全找不到可用 split
    - 非致命警告：data.yaml 中声明的路径在磁盘上不存在（该 split 跳过，继续处理其余 split）
    """
    errors: List[str] = []
    warnings: List[str] = []
    result: List[SplitPaths] = []

    split_specs = [
        ("train", "train", args.train_images, args.train_labels),
        ("val", "valid", args.val_images, args.val_labels),
        ("test", "test", args.test_images, args.test_labels),
    ]

    for yaml_key, out_name, images_override, labels_override in split_specs:
        yaml_images_value = data_yaml.get(yaml_key)
        if images_override is None and not yaml_images_value:
            continue

        from_override = images_override is not None

        if from_override:
            images_dir = images_override.expanduser().resolve()
        else:
            if not isinstance(yaml_images_value, str):
                errors.append(f"{yaml_key} image path in data.yaml must be string")
                continue
            images_dir = resolve_roboflow_relative(data_yaml_path.parent, yaml_images_value)

        if not images_dir.is_dir():
            msg = f"{yaml_key} images directory not found, skipping this split: {images_dir}"
            if from_override:
                errors.append(msg)
            else:
                warnings.append(msg)
            continue

        if labels_override is not None:
            labels_dir = labels_override.expanduser().resolve()
            from_labels_override = True
        else:
            derived = _derive_labels_from_images(images_dir)
            if derived is None:
                errors.append(
                    f"{yaml_key} labels path cannot be derived from images dir: {images_dir}"
                )
                continue
            labels_dir = derived
            from_labels_override = False

        if not labels_dir.is_dir():
            msg = f"{yaml_key} labels directory not found, skipping this split: {labels_dir}"
            if from_labels_override:
                errors.append(msg)
            else:
                warnings.append(msg)
            continue

        result.append(
            SplitPaths(
                split_key=yaml_key,
                split_name=out_name,
                images_dir=images_dir,
                labels_dir=labels_dir,
            )
        )

    if not result:
        errors.append("No valid split found to process")

    return result, errors, warnings


def find_image_file(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
        upper_candidate = images_dir / f"{stem}{ext.upper()}"
        if upper_candidate.is_file():
            return upper_candidate

    # fallback: exact stem match by glob
    matches = list(images_dir.glob(f"{stem}.*"))
    for m in matches:
        if m.is_file():
            return m
    return None


def read_label_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def parse_label_class_id(line: str) -> Optional[int]:
    stripped = line.strip()
    if not stripped:
        return None
    first = stripped.split()[0]
    if first.isdigit():
        return int(first)
    return None


def make_run_dir(output_root: Path, run_name: str, dry_run: bool) -> Path:
    run_dir = output_root / run_name
    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def ensure_split_output_dirs(run_dir: Path, split_name: str, dry_run: bool) -> Tuple[Path, Path]:
    images_out = run_dir / split_name / "images"
    labels_out = run_dir / split_name / "labels"
    if not dry_run:
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
    return images_out, labels_out


def main() -> int:
    args = parse_args()
    merge_enabled = args.merge == "true"

    data_yaml_path = args.data_yaml.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    try:
        data_yaml_obj = load_data_yaml(data_yaml_path)
        names = parse_names(data_yaml_obj.get("names"), data_yaml_obj.get("nc"))
        target_ids, target_names = resolve_targets(args.target_names, args.target_ids, names)
        split_paths, split_errors, split_warnings = resolve_split_paths(args, data_yaml_obj, data_yaml_path)
    except Exception as e:
        _alert(str(e))
        return 1

    for warn in split_warnings:
        _warn(warn)

    if split_errors:
        for err in split_errors:
            _alert(err)
        _alert("Dataset processing is skipped due to invalid path/config")
        return 1

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    dataset_short = _short_dataset_name(data_yaml_path)

    run_specs: List[RunSpec] = []
    if merge_enabled and len(target_ids) > 1:
        run_name = f"{dataset_short}_merge_{timestamp}"
        run_specs.append(
            RunSpec(run_key="merge", run_dir=make_run_dir(output_root, run_name, args.dry_run), categories=set(target_ids), category_names=target_names)
        )
    else:
        for class_id, class_name in zip(target_ids, target_names):
            safe_name = _sanitize_name(class_name)
            run_name = f"{dataset_short}_{safe_name}_{timestamp}"
            run_specs.append(
                RunSpec(
                    run_key=f"class_{class_id}",
                    run_dir=make_run_dir(output_root, run_name, args.dry_run),
                    categories={class_id},
                    category_names=[class_name],
                )
            )

    if args.debug:
        print("Debug details:")
        print(f"  data_yaml: {data_yaml_path}")
        print(f"  output_root: {output_root}")
        print(f"  merge: {merge_enabled}")
        print(f"  dry_run: {args.dry_run}")
        print(f"  target_ids: {target_ids}")
        print(f"  target_names: {target_names}")
        for sp in split_paths:
            print(f"  split {sp.split_name}: images={sp.images_dir} labels={sp.labels_dir}")
        for run in run_specs:
            print(f"  run {run.run_key}: dir={run.run_dir} categories={run.category_names}")
        print("-" * 80)

    run_results: Dict[str, RunResult] = {
        run.run_key: RunResult(
            run_key=run.run_key,
            run_dir=run.run_dir,
            categories=run.category_names,
            copy_stats=Counter(),
        )
        for run in run_specs
    }

    for split in split_paths:
        label_files = sorted([p for p in split.labels_dir.glob("*.txt") if p.is_file()])
        for label_file in label_files:
            lines = read_label_lines(label_file)
            matched_lines_by_run: Dict[str, List[str]] = {run.run_key: [] for run in run_specs}

            for line in lines:
                class_id = parse_label_class_id(line)
                if class_id is None:
                    continue
                for run in run_specs:
                    if class_id in run.categories:
                        matched_lines_by_run[run.run_key].append(line.strip())

            if not any(matched_lines_by_run.values()):
                continue

            stem = label_file.stem
            image_file = find_image_file(split.images_dir, stem)

            for run in run_specs:
                matched_lines = matched_lines_by_run[run.run_key]
                if not matched_lines:
                    continue

                result = run_results[run.run_key]
                result.copy_stats["matched_label_files"] += 1
                result.copy_stats["matched_label_lines"] += len(matched_lines)

                if image_file is None:
                    result.copy_stats["images_missing"] += 1
                    if args.debug:
                        print(f"[debug] image missing for label: {label_file}")
                    continue

                images_out, labels_out = ensure_split_output_dirs(run.run_dir, split.split_name, args.dry_run)
                target_image = images_out / image_file.name
                target_label = labels_out / label_file.name

                if not args.dry_run:
                    shutil.copy2(image_file, target_image)
                    target_label.write_text("\n".join(matched_lines) + "\n", encoding="utf-8")

                result.copy_stats["images_copied"] += 1
                result.copy_stats["labels_written"] += 1

                if args.debug:
                    print(
                        f"[debug] split={split.split_name} run={run.run_key} "
                        f"image={image_file} -> {target_image} label={label_file} -> {target_label}"
                    )

    print("=" * 90)
    print("Roboflow YOLO filter summary")
    print("=" * 90)
    print(f"data_yaml: {data_yaml_path}")
    print(f"dataset_short_name: {dataset_short}")
    print(f"selected_targets: {target_names} ({target_ids})")
    print(f"merge_mode: {merge_enabled}")
    print(f"dry_run: {args.dry_run}")
    print("-" * 90)

    for split in split_paths:
        print(f"split={split.split_name}")
        print(f"  images_dir: {split.images_dir}")
        print(f"  labels_dir: {split.labels_dir}")
        print("-" * 90)

    if args.dry_run:
        print("Dry-run mode: no directory creation, no copy, no label rewrite.")

    for run in run_specs:
        stats = run_results[run.run_key].copy_stats
        print(f"output run: {run.run_key}")
        print(f"  categories: {run.category_names}")
        print(f"  output_directory: {run.run_dir}")
        print(f"  matched_label_files: {stats['matched_label_files']}")
        print(f"  matched_label_lines: {stats['matched_label_lines']}")
        print(f"  images_copied: {stats['images_copied']}")
        print(f"  images_missing: {stats['images_missing']}")
        print(f"  labels_written: {stats['labels_written']}")
        print("-" * 90)

    if args.print_output_dir:
        for run in run_specs:
            print(f"OUTPUT_DIR:{run.run_dir}")

    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
