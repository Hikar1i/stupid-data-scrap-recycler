#!/usr/bin/env python3
"""将 YOLO 矩形框标签 .txt 文件转换为 labelme JSON 格式。

类别映射可通过以下两种方式提供（二选一）：
  --mapping       "0:cat,2:toy"        （内联格式，支持含空格的类别名）
  --classes-file  /path/classes.txt   （每行一个类别名，行号即 ID，从 0 开始）

输出 JSON 默认写入 images/ 目录（与图像共存），指定 --output-dir 时按源目录结构镜像输出。
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image as PILImage
except ImportError:
    print(
        "错误：读取图像尺寸需要 Pillow 库，"
        "请安装：pip install Pillow",
        file=sys.stderr,
    )
    sys.exit(1)


LABELME_VERSION = "3.3.0-beta"


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 YOLO 矩形框标签 .txt 文件转换为 labelme JSON 格式。"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help=(
            "YOLO 数据集根目录（含 train/val/test 子目录）"
            "或单个 split 目录（直接含 images/ 和 labels/）。"
        ),
    )

    mapping_group = parser.add_mutually_exclusive_group(required=True)
    mapping_group.add_argument(
        "--mapping",
        type=str,
        help=(
            "类别 ID 到类别名称映射，例如 '0:cat,2:toy' 或 '6:No helmet,7:Safety Vest'。"
            "逗号分隔每个映射对，类别名内允许含空格。"
        ),
    )
    mapping_group.add_argument(
        "--classes-file",
        type=Path,
        help="classes.txt 路径，每行一个类别名，第一行对应 ID 0。",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "labelme JSON 输出目录。"
            "未指定时 JSON 写入每个 images/ 目录（与图像共存）。"
            "指定时按源目录结构输出到 labelme_<时间戳>/ 子目录。"
        ),
    )
    parser.add_argument(
        "--overwrite",
        choices=["true", "false"],
        default="false",
        help="为 false（默认）时跳过已存在的 JSON 文件。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式，不创建任何文件。",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="打印每个文件的转换路径详情。",
    )
    parser.add_argument(
        "--print-output-dir",
        action="store_true",
        help="在末尾打印 OUTPUT_DIR:<路径> 行，供 pipeline wrapper 捕获。",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 映射解析工具函数
# ---------------------------------------------------------------------------

def parse_mapping_inline(raw: str) -> Dict[int, str]:
    """解析内联映射字符串 '0:cat,6:No helmet,7:Safety Vest' → {0: 'cat', 6: 'No helmet', 7: 'Safety Vest'}

    策略：只在絺号后紧跟数字+冒号时切割，保留类别名内部空格。
    """
    # 仅在逗号后紧跟 <数字>: 时切割
    parts = re.split(r",(?=\d+:)", raw)
    mapping: Dict[int, str] = {}
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d+):(.+)$", part)
        if not m:
            raise ValueError(f"映射格式不合法（期望 '整数:名称'）：'{part}'")
        cid = int(m.group(1))
        name = m.group(2).strip()
        if not name:
            raise ValueError(f"ID {cid} 对应的类别名为空")
        if cid in mapping:
            raise ValueError(f"映射中存在重复的类别 ID：{cid}")
        mapping[cid] = name
    if not mapping:
        raise ValueError("映射为空。")
    return mapping


def parse_mapping_from_classes_file(path: Path) -> Dict[int, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    mapping: Dict[int, str] = {}
    for idx, line in enumerate(lines):
        name = line.strip()
        if name:
            mapping[idx] = name
    if not mapping:
        raise ValueError(f"classes-file 为空或无有效内容：{path}")
    return mapping


# ---------------------------------------------------------------------------
# 数据集目录结构探测
# ---------------------------------------------------------------------------

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def find_split_pairs(source_dir: Path) -> List[Tuple[Path, Path]]:
    """
    在 source_dir 下找到所有 (images_dir, labels_dir) 配对。
    有效配对要求 images/ 和 labels/ 均存在且互为兄弟目录。

    额外支持 source_dir 本身即为 labels 目录的情形（例如 pipeline reindex
    阶段输出的 labels_remapping_<ts>/ 目录）：若其父目录下存在同级 images/
    目录，则直接将该对作为唯一配对返回。
    """
    pairs: List[Tuple[Path, Path]] = []
    seen_images: set = set()

    # 检测 source_dir 自身是否为 labels 目录（父目录下有同级 images/）
    images_sibling = source_dir.parent / "images"
    if images_sibling.is_dir():
        seen_images.add(images_sibling)
        pairs.append((images_sibling, source_dir))

    for labels_dir in sorted(source_dir.rglob("labels")):
        if not labels_dir.is_dir():
            continue
        images_dir = labels_dir.parent / "images"
        if images_dir.is_dir() and images_dir not in seen_images:
            seen_images.add(images_dir)
            pairs.append((images_dir, labels_dir))

    if not pairs:
        raise FileNotFoundError(
            f"在 {source_dir} 下未找到匹配的 images/+labels/ 目录对"
        )
    return pairs


def match_image_for_label(label_file: Path, images_dir: Path) -> Optional[Path]:
    """查找与标签文件同名的图像文件。"""
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / (label_file.stem + suffix)
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# 构建 labelme JSON
# ---------------------------------------------------------------------------

def read_image_size(image_path: Path) -> Tuple[int, int]:
    """使用 Pillow 读取图像尺寸，返回 (width, height)。"""
    with PILImage.open(image_path) as img:
        return img.size  # (width, height)


def yolo_bbox_to_labelme_points(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> List[List[float]]:
    """将 YOLO 归一化矩形框转为 4 个角点像素坐标（左上→右上→右下→左下）。"""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ]


def build_labelme_json(
    image_path: Path,
    label_lines: List[str],
    class_map: Dict[int, str],
    unknown_ids: set,
) -> dict:
    img_w, img_h = read_image_size(image_path)
    shapes = []

    for line in label_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue

        if cid not in class_map:
            unknown_ids.add(cid)
            label_name = "unknown"
        else:
            label_name = class_map[cid]

        points = yolo_bbox_to_labelme_points(cx, cy, bw, bh, img_w, img_h)
        shapes.append(
            {
                "label": label_name,
                "score": None,
                "points": points,
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            }
        )

    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
    }


# ---------------------------------------------------------------------------
# 输出路径解析
# ---------------------------------------------------------------------------

def resolve_json_output_path(
    label_file: Path,
    images_dir: Path,
    output_dir: Optional[Path],
    source_dir: Path,
    timestamp: str,
) -> Path:
    if output_dir is None:
        # 默认：将 JSON 写入 images/ 目录（与图像共存）
        return images_dir / (label_file.stem + ".json")

    # 镜像源目录结构：output_dir/<相对split路径>/labelme_<ts>/<文件名>.json
    try:
        rel_split = label_file.parent.parent.relative_to(source_dir)
    except ValueError:
        rel_split = Path(label_file.parent.parent.name)
    return output_dir / rel_split / f"labelme_{timestamp}" / (label_file.stem + ".json")


# ---------------------------------------------------------------------------
# 核心处理逻辑
# ---------------------------------------------------------------------------

def process_split_pair(
    images_dir: Path,
    labels_dir: Path,
    class_map: Dict[int, str],
    output_dir: Optional[Path],
    source_dir: Path,
    overwrite: bool,
    dry_run: bool,
    debug: bool,
    timestamp: str,
) -> Tuple[dict, set]:
    """处理一个 (images_dir, labels_dir) 配对，返回 (统计字典, 未知 ID 集合)。"""
    stats = {
        "label_files": 0,
        "converted": 0,
        "skipped_no_image": 0,
        "skipped_exists": 0,
        "skipped_empty": 0,
        "error": 0,
    }
    unknown_ids: set = set()

    label_files = sorted(labels_dir.glob("*.txt"))
    stats["label_files"] = len(label_files)

    for lf in label_files:
        image_path = match_image_for_label(lf, images_dir)
        if image_path is None:
            stats["skipped_no_image"] += 1
            if debug:
                print(f"  [NO IMAGE] {lf.name}")
            continue

        out_json = resolve_json_output_path(lf, images_dir, output_dir, source_dir, timestamp)

        if out_json.exists() and not overwrite:
            stats["skipped_exists"] += 1
            if debug:
                print(f"  [EXISTS SKIP] {out_json}")
            continue

        lines = lf.read_text(encoding="utf-8").splitlines()
        non_empty = [l for l in lines if l.strip()]
        if not non_empty:
            stats["skipped_empty"] += 1
            if debug:
                print(f"  [EMPTY LABEL] {lf.name}")
            continue

        try:
            data = build_labelme_json(image_path, lines, class_map, unknown_ids)
        except Exception as e:
            stats["error"] += 1
            print(f"  [ERROR] {lf}: {e}", file=sys.stderr)
            continue

        if debug:
            print(f"  [CONVERT] {lf.name} → {out_json}")

        if not dry_run:
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        stats["converted"] += 1

    return stats, unknown_ids


# ---------------------------------------------------------------------------
# 入口函数
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if not args.source_dir.is_dir():
        print(f"错误：source-dir 不存在：{args.source_dir}", file=sys.stderr)
        return 1

    # 构建类别映射
    try:
        if args.mapping:
            class_map = parse_mapping_inline(args.mapping)
        else:
            if not args.classes_file.is_file():
                print(f"错误：classes-file 不存在：{args.classes_file}", file=sys.stderr)
                return 1
            class_map = parse_mapping_from_classes_file(args.classes_file)
    except ValueError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1

    overwrite = args.overwrite == "true"
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    try:
        pairs = find_split_pairs(args.source_dir)
    except FileNotFoundError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1

    print("=" * 90)
    print("YOLO → labelme 格式转换")
    print("=" * 90)
    print(f"source_dir:    {args.source_dir}")
    print(f"class_map:     {class_map}")
    print(f"output_dir:    {args.output_dir or '(写入 images/ 目录)' }")
    print(f"overwrite:     {overwrite}")
    print(f"dry_run:       {args.dry_run}")
    print(f"找到的 split 配对数: {len(pairs)}")
    for img_d, lbl_d in pairs:
        print(f"  images: {img_d}")
        print(f"  labels: {lbl_d}")
    print("-" * 90)

    all_unknown_ids: set = set()
    all_output_dirs: List[Path] = []

    for images_dir, labels_dir in pairs:
        stats, unknown_ids = process_split_pair(
            images_dir=images_dir,
            labels_dir=labels_dir,
            class_map=class_map,
            output_dir=args.output_dir,
            source_dir=args.source_dir,
            overwrite=overwrite,
            dry_run=args.dry_run,
            debug=args.debug,
            timestamp=timestamp,
        )
        all_unknown_ids.update(unknown_ids)

        if args.output_dir is not None:
            try:
                rel_split = labels_dir.parent.relative_to(args.source_dir)
            except ValueError:
                rel_split = Path(labels_dir.parent.name)
            out_root = args.output_dir / rel_split / f"labelme_{timestamp}"
        else:
            out_root = images_dir
        all_output_dirs.append(out_root)

        mode_tag = "[DRY-RUN]" if args.dry_run else "[OUTPUT]"
        print(f"{mode_tag} split: {images_dir.parent}")
        print(f"  output_location:    {out_root}")
        print(f"  label_files:        {stats['label_files']}")
        print(f"  converted:          {stats['converted']}")
        print(f"  skipped_no_image:   {stats['skipped_no_image']}")
        print(f"  skipped_exists:     {stats['skipped_exists']}")
        print(f"  skipped_empty:      {stats['skipped_empty']}")
        print(f"  errors:             {stats['error']}")
        print("-" * 90)

    if all_unknown_ids:
        print("!" * 90)
        print("警告：检测到未知类别 ID！")
        print(
            f"  标签文件中以下 ID 在映射中无对应关系：{sorted(all_unknown_ids)}"
        )
        print(
            "  这些标签已在输出 JSON 中写入为 'unknown'。\n"
            "  解决方法：将缺失的 ID 添加到 --mapping 或 --classes-file 并重新运行\n"
            "  （使用 --overwrite true 覆盖已有输出）。"
        )
        print("!" * 90)

    if args.print_output_dir:
        for out_dir in all_output_dirs:
            print(f"OUTPUT_DIR:{out_dir}")

    print("=" * 90)
    return 0


if __name__ == "__main__":
    sys.exit(main())
