#!/usr/bin/env python3
import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class FilePair:
    json_prefix: str
    image_source: Path
    image_name: str
    label_source: Path
    label_name: str


@dataclass
class JsonStats:
    json_path: Path
    resolved_category_ids: List[int]
    resolved_category_names: List[str]
    matched_annotations: int
    unique_image_ids: int
    matched_file_names: int


@dataclass
class RunSpec:
    run_key: str
    run_dir: Path
    category_ids: Set[int]
    category_names: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter COCO annotations by category and copy matched images/labels "
            "to timestamped target directory."
        )
    )

    category_group = parser.add_mutually_exclusive_group(required=True)
    category_group.add_argument(
        "--category-ids",
        type=str,
        help="Target category ids, comma-separated, e.g. 1,2",
    )
    category_group.add_argument(
        "--category-names",
        type=str,
        help="Target category names, comma-separated, e.g. boom,fence",
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--json-file", type=Path, help="Single COCO JSON file")
    source_group.add_argument(
        "--json-dir", type=Path, help="Directory containing COCO JSON files"
    )

    parser.add_argument(
        "--label-root",
        type=Path,
        required=True,
        help="Label base path used to build <label_root>/<image_parent>/<image_stem>.txt",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory to create <run_name>/<json_prefix>/{images,labels}",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=(
            "Optional image base path. If omitted, relative file_name in JSON is "
            "resolved against the JSON file's parent directory."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview output paths and stats without creating directories or copying files",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed per-run/per-json debug details",
    )
    parser.add_argument(
        "--merge",
        choices=["true", "false"],
        default="false",
        help=(
            "Multi-category output mode. true: merge selected categories into one directory; "
            "false: one directory per category"
        ),
    )
    parser.add_argument(
        "--print-output-dir",
        action="store_true",
        help="Print OUTPUT_DIR:<path> lines at the end (used by pipeline wrappers).",
    )

    return parser.parse_args()


def _sanitize_name(name: str) -> str:
    cleaned = name.strip().replace("/", "_").replace("\\", "_").replace(" ", "-")
    return cleaned if cleaned else "unknown"


def _parse_multi_items(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_target_inputs(args: argparse.Namespace) -> Tuple[List[int], List[str], bool]:
    if args.category_ids:
        raw_ids = _parse_multi_items(args.category_ids)
        if not raw_ids:
            raise ValueError("category-ids is empty")
        parsed_ids: List[int] = []
        for item in raw_ids:
            if not item.isdigit():
                raise ValueError(f"category id is not integer: {item}")
            parsed_ids.append(int(item))
        return sorted(set(parsed_ids)), [], True

    raw_names = _parse_multi_items(args.category_names)
    if not raw_names:
        raise ValueError("category-names is empty")
    return [], sorted(set(raw_names)), False


def collect_json_files(json_file: Optional[Path], json_dir: Optional[Path]) -> List[Path]:
    if json_file is not None:
        if not json_file.is_file():
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        return [json_file]

    if json_dir is None or not json_dir.is_dir():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    files = sorted([p for p in json_dir.glob("*.json") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No JSON files found under: {json_dir}")
    return files


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_target_map_for_json(
    categories: Iterable[dict],
    target_ids_input: List[int],
    target_names_input: List[str],
    use_ids: bool,
) -> Dict[int, str]:
    resolved: Dict[int, str] = {}
    if use_ids:
        id_to_name = {c.get("id"): str(c.get("name")) for c in categories}
        for cid in target_ids_input:
            if cid in id_to_name:
                resolved[cid] = id_to_name[cid]
        return resolved

    name_to_id: Dict[str, int] = {}
    for c in categories:
        cname = c.get("name")
        cid = c.get("id")
        if isinstance(cname, str) and isinstance(cid, int):
            name_to_id[cname] = cid
    for cname in target_names_input:
        cid = name_to_id.get(cname)
        if cid is not None:
            resolved[cid] = cname
    return resolved


def find_file_names_by_categories(data: dict, target_category_ids: Set[int]) -> Tuple[int, Set[int], List[str]]:
    matched_image_ids: Set[int] = set()
    matched_annotations = 0

    for ann in data.get("annotations", []):
        if ann.get("category_id") in target_category_ids:
            image_id = ann.get("image_id")
            if image_id is not None:
                matched_image_ids.add(image_id)
                matched_annotations += 1

    file_names: List[str] = []
    for img in data.get("images", []):
        if img.get("id") in matched_image_ids:
            name = img.get("file_name")
            if isinstance(name, str) and name:
                file_names.append(name)

    return matched_annotations, matched_image_ids, file_names


def build_file_pairs(
    json_path: Path,
    json_prefix: str,
    file_names: Iterable[str],
    label_root: Path,
    image_root: Optional[Path],
) -> List[FilePair]:
    pairs: List[FilePair] = []
    for file_name in file_names:
        file_path = Path(file_name)
        image_name = file_path.name
        label_parent = file_path.parent.name if file_path.parent.name not in ("", ".") else ""
        label_name = f"{file_path.stem}.txt"

        if file_path.is_absolute():
            image_source = file_path
        else:
            base = image_root if image_root is not None else json_path.parent
            image_source = base / file_path

        if label_parent:
            label_source = label_root / label_parent / label_name
        else:
            label_source = label_root / label_name
        pairs.append(
            FilePair(
                json_prefix=json_prefix,
                image_source=image_source,
                image_name=image_name,
                label_source=label_source,
                label_name=label_name,
            )
        )
    return pairs


def ensure_run_dir(output_root: Path, run_name: str, create_dirs: bool) -> Path:
    run_dir = output_root / run_name
    if create_dirs:
        run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def ensure_json_output_dirs(run_dir: Path, json_prefix: str, create_dirs: bool) -> Tuple[Path, Path]:
    json_dir = run_dir / json_prefix
    images_dir = json_dir / "images"
    labels_dir = json_dir / "labels"
    if create_dirs:
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def parse_label_class_id(line: str) -> Optional[int]:
    stripped = line.strip()
    if not stripped:
        return None
    first = stripped.split()[0]
    if first.isdigit():
        return int(first)
    return None


def read_filtered_label_lines(label_path: Path, allowed_category_ids: Set[int]) -> List[str]:
    lines = label_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if parse_label_class_id(line) in allowed_category_ids]


def coco_ids_to_yolo_label_ids(coco_ids: Set[int]) -> Set[int]:
    return {cid - 1 for cid in coco_ids if cid > 0}


def copy_pairs_with_label_filter(
    pairs: List[FilePair],
    images_dir: Path,
    labels_dir: Path,
    allowed_category_ids: Set[int],
    dry_run: bool,
) -> Counter:
    stats = Counter()
    used_image_names: Dict[str, Path] = {}
    used_label_names: Dict[str, Path] = {}

    for pair in pairs:
        prev_image = used_image_names.get(pair.image_name)
        if prev_image is not None and prev_image != pair.image_source:
            stats["image_name_conflicts"] += 1
            continue
        used_image_names[pair.image_name] = pair.image_source

        prev_label = used_label_names.get(pair.label_name)
        if prev_label is not None and prev_label != pair.label_source:
            stats["label_name_conflicts"] += 1
            continue
        used_label_names[pair.label_name] = pair.label_source

        if not (pair.image_source.exists() and pair.image_source.is_file()):
            stats["images_missing"] += 1
            continue

        if not (pair.label_source.exists() and pair.label_source.is_file()):
            stats["labels_missing"] += 1
            continue

        filtered_lines = read_filtered_label_lines(pair.label_source, allowed_category_ids)
        if not filtered_lines:
            stats["labels_filtered_empty"] += 1
            continue

        if not dry_run:
            shutil.copy2(pair.image_source, images_dir / pair.image_name)
            (labels_dir / pair.label_name).write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")

        stats["images_copied"] += 1
        stats["labels_written"] += 1
        stats["matched_label_lines"] += len(filtered_lines)

    return stats


def main() -> None:
    args = parse_args()
    merge_enabled = args.merge == "true"
    target_ids_input, target_names_input, use_ids = parse_target_inputs(args)
    json_files = collect_json_files(args.json_file, args.json_dir)

    json_pairs: Dict[Path, List[FilePair]] = {}
    json_resolved: Dict[Path, Dict[int, str]] = {}
    json_stats: List[JsonStats] = []

    for json_path in json_files:
        data = load_json(json_path)
        resolved_map = resolve_target_map_for_json(
            data.get("categories", []),
            target_ids_input,
            target_names_input,
            use_ids,
        )
        resolved_ids = set(resolved_map.keys())
        matched_annotations, image_ids, file_names = find_file_names_by_categories(data, resolved_ids)
        pairs = build_file_pairs(
            json_path=json_path,
            json_prefix=json_path.stem,
            file_names=file_names,
            label_root=args.label_root,
            image_root=args.image_root,
        )
        json_pairs[json_path] = pairs
        json_resolved[json_path] = resolved_map
        json_stats.append(
            JsonStats(
                json_path=json_path,
                resolved_category_ids=sorted(resolved_map.keys()),
                resolved_category_names=[resolved_map[k] for k in sorted(resolved_map.keys())],
                matched_annotations=matched_annotations,
                unique_image_ids=len(image_ids),
                matched_file_names=len(file_names),
            )
        )

    all_resolved_id_to_name: Dict[int, str] = {}
    for m in json_resolved.values():
        for cid, cname in m.items():
            all_resolved_id_to_name[cid] = cname

    selected_targets: List[Tuple[int, str]] = []
    if use_ids:
        for cid in target_ids_input:
            cname = all_resolved_id_to_name.get(cid, f"category_{cid}")
            selected_targets.append((cid, cname))
    else:
        for cname in target_names_input:
            matched_id = None
            for cid, resolved_name in all_resolved_id_to_name.items():
                if resolved_name == cname:
                    matched_id = cid
                    break
            if matched_id is None:
                matched_id = -1
            selected_targets.append((matched_id, cname))

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_specs: List[RunSpec] = []
    if merge_enabled and len(selected_targets) > 1:
        run_specs.append(
            RunSpec(
                run_key="merge",
                run_dir=ensure_run_dir(args.output_root, f"merge_{timestamp}", create_dirs=not args.dry_run),
                category_ids=set([cid for cid, _ in selected_targets if cid >= 0]),
                category_names=[name for _, name in selected_targets],
            )
        )
    else:
        for cid, cname in selected_targets:
            run_specs.append(
                RunSpec(
                    run_key=cname,
                    run_dir=ensure_run_dir(
                        args.output_root,
                        f"{_sanitize_name(cname)}_{timestamp}",
                        create_dirs=not args.dry_run,
                    ),
                    category_ids={cid} if cid >= 0 else set(),
                    category_names=[cname],
                )
            )

    run_stats: Dict[str, Counter] = {run.run_key: Counter() for run in run_specs}
    run_coco_ids: Dict[str, Set[int]] = {run.run_key: set() for run in run_specs}
    run_yolo_label_ids: Dict[str, Set[int]] = {run.run_key: set() for run in run_specs}

    for run in run_specs:
        for json_path in json_files:
            resolved_map = json_resolved[json_path]
            if run.run_key == "merge":
                allowed_ids = set(resolved_map.keys())
            else:
                allowed_ids = {cid for cid, name in resolved_map.items() if name in run.category_names}

            if not allowed_ids:
                continue

            allowed_label_ids = coco_ids_to_yolo_label_ids(allowed_ids)
            run_coco_ids[run.run_key].update(allowed_ids)
            run_yolo_label_ids[run.run_key].update(allowed_label_ids)

            filtered_pairs = [p for p in json_pairs[json_path]]
            images_dir, labels_dir = ensure_json_output_dirs(
                run.run_dir,
                json_path.stem,
                create_dirs=not args.dry_run,
            )
            stats = copy_pairs_with_label_filter(
                pairs=filtered_pairs,
                images_dir=images_dir,
                labels_dir=labels_dir,
                allowed_category_ids=allowed_label_ids,
                dry_run=args.dry_run,
            )
            run_stats[run.run_key].update(stats)

    print("=" * 90)
    print("COCO category filter summary")
    print("=" * 90)
    print(f"processed_json_files: {len(json_files)}")
    print(f"input_mode: {'category_ids' if use_ids else 'category_names'}")
    print(f"merge_mode: {merge_enabled}")
    print(f"dry_run: {args.dry_run}")
    print("!" * 90)
    print("ID mapping notice: COCO category_id is treated as 1-based, YOLO label id as 0-based.")
    print("Label filtering rule: yolo_label_id = coco_category_id - 1")
    if args.dry_run:
        print("Current mode: dry-run (mapping is still applied when computing filtering statistics).")
    else:
        print("Current mode: execute copy/write (mapping is applied to actual label-line filtering).")
    print("!" * 90)
    print("-" * 90)

    for item in json_stats:
        print(f"json: {item.json_path}")
        print(f"  resolved_category_ids: {item.resolved_category_ids}")
        print(f"  resolved_category_names: {item.resolved_category_names}")
        print(f"  matched_annotations: {item.matched_annotations}")
        print(f"  unique_image_ids: {item.unique_image_ids}")
        print(f"  matched_file_names: {item.matched_file_names}")
        print("-" * 90)

    if args.debug:
        print("Debug details:")
        for run in run_specs:
            print(f"  run={run.run_key}")
            print(f"    run_dir: {run.run_dir}")
            print(f"    category_names: {run.category_names}")
            for json_path in json_files:
                print(f"    json={json_path}")
                for pair in json_pairs[json_path]:
                    print(f"      file_name: {pair.image_source}")
                    print(f"      label_path: {pair.label_source}")
        print("-" * 90)

    if args.dry_run:
        print("Dry-run mode: no directory creation, no copy, no label rewrite.")

    for run in run_specs:
        stats = run_stats[run.run_key]
        print(f"output run: {run.run_key}")
        print(f"  categories: {run.category_names}")
        print(f"  output_directory: {run.run_dir}")
        print(f"  coco_category_ids_for_annotations: {sorted(run_coco_ids[run.run_key])}")
        print(f"  yolo_label_ids_for_filtering: {sorted(run_yolo_label_ids[run.run_key])}")
        print(f"  matched_label_lines: {stats['matched_label_lines']}")
        print(f"  images_copied: {stats['images_copied']}")
        print(f"  images_missing: {stats['images_missing']}")
        print(f"  labels_written: {stats['labels_written']}")
        print(f"  labels_missing: {stats['labels_missing']}")
        print(f"  labels_filtered_empty: {stats['labels_filtered_empty']}")
        print(f"  image_name_conflicts: {stats['image_name_conflicts']}")
        print(f"  label_name_conflicts: {stats['label_name_conflicts']}")
        print("-" * 90)

    if args.print_output_dir:
        for run in run_specs:
            print(f"OUTPUT_DIR:{run.run_dir}")

    print("=" * 90)


if __name__ == "__main__":
    main()
