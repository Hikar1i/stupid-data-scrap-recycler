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
    image_source: Path
    image_name: str
    label_source: Path
    label_name: str


@dataclass
class JsonStats:
    json_path: Path
    category_id: Optional[int]
    category_name: Optional[str]
    matched_annotations: int
    unique_image_ids: int
    matched_file_names: int


@dataclass
class CopyRunResult:
    run_prefix: str
    source_jsons: List[Path]
    run_dir: Path
    images_dir: Path
    labels_dir: Path
    total_pairs: int
    copy_stats: Counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter COCO annotations by category and copy matched images/labels "
            "to timestamped target directory."
        )
    )

    category_group = parser.add_mutually_exclusive_group(required=True)
    category_group.add_argument("--category-id", type=int, help="Target category_id")
    category_group.add_argument("--category-name", type=str, help="Target category_name")

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
        help="Root directory to create <category_name>_<timestamp>/images and labels",
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
        help="Print each matched file_name and resolved label absolute path",
    )
    parser.add_argument(
        "--merge",
        choices=["true", "false"],
        default="false",
        help=(
            "Only for --json-dir. true: merge all JSON copy results into one output directory; "
            "false: create one output directory per JSON file"
        ),
    )

    return parser.parse_args()


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


def resolve_category(
    categories: Iterable[dict], category_id: Optional[int], category_name: Optional[str]
) -> Tuple[Optional[int], Optional[str]]:
    if category_id is not None:
        for c in categories:
            if c.get("id") == category_id:
                return category_id, c.get("name")
        return None, None

    target_name = category_name
    for c in categories:
        if c.get("name") == target_name:
            return c.get("id"), c.get("name")
    return None, None


def find_file_names_by_category(data: dict, target_category_id: int) -> Tuple[int, Set[int], List[str]]:
    matched_image_ids: Set[int] = set()
    matched_annotations = 0

    for ann in data.get("annotations", []):
        if ann.get("category_id") == target_category_id:
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
                image_source=image_source,
                image_name=image_name,
                label_source=label_source,
                label_name=label_name,
            )
        )
    return pairs


def make_output_dirs(
    output_root: Path,
    run_prefix: str,
    category_name: str,
    create_dirs: bool = True,
) -> Tuple[Path, Path, Path]:
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = output_root / f"{run_prefix}_{category_name}_{timestamp}"
    images_dir = run_dir / "images"
    labels_dir = run_dir / "labels"
    if create_dirs:
        images_dir.mkdir(parents=True, exist_ok=False)
        labels_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, images_dir, labels_dir


def copy_pairs(pairs: List[FilePair], images_dir: Path, labels_dir: Path) -> Counter:
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

        if pair.image_source.exists() and pair.image_source.is_file():
            shutil.copy2(pair.image_source, images_dir / pair.image_name)
            stats["images_copied"] += 1
        else:
            stats["images_missing"] += 1

        if pair.label_source.exists() and pair.label_source.is_file():
            shutil.copy2(pair.label_source, labels_dir / pair.label_name)
            stats["labels_copied"] += 1
        else:
            stats["labels_missing"] += 1

    return stats


def main() -> None:
    args = parse_args()
    json_files = collect_json_files(args.json_file, args.json_dir)
    merge_enabled = args.merge == "true"

    pairs_by_json: Dict[Path, List[FilePair]] = {}
    json_stats: List[JsonStats] = []
    selected_category_name: Optional[str] = args.category_name

    for json_path in json_files:
        data = load_json(json_path)

        resolved_id, resolved_name = resolve_category(
            data.get("categories", []), args.category_id, args.category_name
        )
        if resolved_id is None:
            json_stats.append(
                JsonStats(
                    json_path=json_path,
                    category_id=None,
                    category_name=None,
                    matched_annotations=0,
                    unique_image_ids=0,
                    matched_file_names=0,
                )
            )
            continue

        selected_category_name = selected_category_name or resolved_name or f"category_{resolved_id}"

        matched_annotations, image_ids, file_names = find_file_names_by_category(data, resolved_id)
        pairs = build_file_pairs(
            json_path=json_path,
            file_names=file_names,
            label_root=args.label_root,
            image_root=args.image_root,
        )
        pairs_by_json[json_path] = pairs

        json_stats.append(
            JsonStats(
                json_path=json_path,
                category_id=resolved_id,
                category_name=resolved_name,
                matched_annotations=matched_annotations,
                unique_image_ids=len(image_ids),
                matched_file_names=len(file_names),
            )
        )

    for json_path in json_files:
        pairs_by_json.setdefault(json_path, [])

    if selected_category_name is None:
        selected_category_name = f"category_{args.category_id}"

    run_specs: List[Tuple[str, List[Path], List[FilePair]]] = []
    if args.json_dir is not None:
        if merge_enabled:
            merged_pairs: List[FilePair] = []
            for json_path in json_files:
                merged_pairs.extend(pairs_by_json[json_path])
            run_specs.append(("merge", json_files, merged_pairs))
        else:
            for json_path in json_files:
                run_specs.append((json_path.stem, [json_path], pairs_by_json[json_path]))
    else:
        json_path = json_files[0]
        run_specs.append((json_path.stem, [json_path], pairs_by_json[json_path]))

    run_results: List[CopyRunResult] = []
    for run_prefix, source_jsons, run_pairs in run_specs:
        run_dir, images_dir, labels_dir = make_output_dirs(
            args.output_root,
            run_prefix,
            selected_category_name,
            create_dirs=not args.dry_run,
        )

        if args.dry_run:
            copy_stats = Counter()
        else:
            copy_stats = copy_pairs(run_pairs, images_dir, labels_dir)

        run_results.append(
            CopyRunResult(
                run_prefix=run_prefix,
                source_jsons=source_jsons,
                run_dir=run_dir,
                images_dir=images_dir,
                labels_dir=labels_dir,
                total_pairs=len(run_pairs),
                copy_stats=copy_stats,
            )
        )

    print("=" * 80)
    print("COCO category filter summary")
    print("=" * 80)
    print(f"Processed JSON files: {len(json_files)}")
    print(f"Selected category: {selected_category_name}")
    print(f"Merge mode: {merge_enabled}")
    print(f"Dry run: {args.dry_run}")
    print("-" * 80)

    for item in json_stats:
        print(f"JSON: {item.json_path}")
        if item.category_id is None:
            print("  category: not found")
        else:
            print(f"  category_id: {item.category_id}")
            print(f"  category_name: {item.category_name}")
            print(f"  matched_annotations: {item.matched_annotations}")
            print(f"  unique_image_ids: {item.unique_image_ids}")
            print(f"  matched_file_names: {item.matched_file_names}")
        print("-" * 80)

    if args.debug:
        print("Debug details:")
        for run in run_results:
            print(f"  run_prefix: {run.run_prefix}")
            for pair in [p for spec in run_specs if spec[0] == run.run_prefix for p in spec[2]]:
                print(f"    file_name: {pair.image_source}")
                print(f"    label_path: {pair.label_source}")
        print("-" * 80)

    if args.dry_run:
        print("Dry-run mode: copy is skipped. The files above would be copied to the output directories shown.")

    for run in run_results:
        print(f"Copy result ({run.run_prefix}):")
        print(f"  source_jsons: {[str(p) for p in run.source_jsons]}")
        print(f"  output_directory: {run.run_dir}")
        print(f"  output_images_directory: {run.images_dir}")
        print(f"  output_labels_directory: {run.labels_dir}")
        print(f"  total_pairs: {run.total_pairs}")
        print(f"  images_copied: {run.copy_stats['images_copied']}")
        print(f"  images_missing: {run.copy_stats['images_missing']}")
        print(f"  labels_copied: {run.copy_stats['labels_copied']}")
        print(f"  labels_missing: {run.copy_stats['labels_missing']}")
        print(f"  image_name_conflicts: {run.copy_stats['image_name_conflicts']}")
        print(f"  label_name_conflicts: {run.copy_stats['label_name_conflicts']}")
        print("-" * 80)

    print("=" * 80)


if __name__ == "__main__":
    main()
