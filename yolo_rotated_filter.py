#!/usr/bin/env python3
"""筛除旋转增强导致黑边明显的图像。

扫描 --scan-dir 下所有包含 images/ 子目录的 split 目录，按“外环黑边占比”判定
旋转增强图像，并将其从原 images/、labels/ 中移出，保存到 --output-root 下对应
层级的 images_rotated_*/labels_rotated_* 目录。

判定规则（默认）：
- 外环宽度：图像宽高的 25%
- 黑色阈值：灰度 <= 10 视为黑色像素
- 判定阈值：外环区域黑色像素占比 >= 30%
"""

import argparse
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
from PIL import Image


IMG_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class SplitResult:
    split_dir: Path
    out_images_rotated_dir: Path
    out_labels_rotated_dir: Optional[Path]
    total_images: int
    rotated_count: int
    kept_count: int
    moved_json_count: int
    moved_label_count: int


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "筛除旋转增强导致黑边明显的图像。\n"
            "扫描 --scan-dir 下所有包含 images/ 子目录的 split 目录，\n"
            "将判定为旋转图像的图片与对应标注移出到 --output-root。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scan-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="扫描根目录，递归查找所有包含 images/ 子目录的 split 目录",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        metavar="DIR",
        help=(
            "旋转图像输出根目录，保留 scan-dir 下原始层级，"
            "在每个 split 下创建 images_rotated_*/labels_rotated_*"
        ),
    )
    parser.add_argument(
        "--edge-width-percent",
        type=float,
        default=25.0,
        metavar="FLOAT",
        help="外环检测宽度百分比（默认 25）",
    )
    parser.add_argument(
        "--black-area-percent",
        type=float,
        default=30.0,
        metavar="FLOAT",
        help="外环黑色像素占比阈值百分比（默认 30）",
    )
    parser.add_argument(
        "--black-pixel-threshold",
        type=int,
        default=10,
        metavar="INT",
        help="灰度黑色像素阈值（<= 该值视为黑色，默认 10）",
    )
    parser.add_argument(
        "--timestamp-suffix",
        choices=["true", "false"],
        default="true",
        help="输出目录是否附加时间戳后缀（默认 true）",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不移动文件")
    parser.add_argument("--debug", action="store_true", help="打印逐文件判定信息")
    parser.add_argument(
        "--print-output-dir",
        action="store_true",
        help="处理完成后逐行打印 OUTPUT_DIR:<split_dir>，供 pipeline 捕获（可选）",
    )
    return parser.parse_args(argv)


def scan_split_dirs(scan_dir: Path) -> List[Path]:
    """扫描 scan_dir 下所有直接包含 images/ 子目录的目录（即 split 目录）。"""
    split_dirs: List[Path] = []
    for images_subdir in sorted(scan_dir.rglob("images")):
        if images_subdir.is_dir():
            split_dirs.append(images_subdir.parent)

    seen: Set[Path] = set()
    result: List[Path] = []
    for split_dir in split_dirs:
        if split_dir not in seen:
            seen.add(split_dir)
            result.append(split_dir)
    return result


def find_image_paths(images_dir: Path) -> List[Path]:
    """获取 images 目录下全部图像文件（递归）。"""
    return sorted(
        path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMG_EXTS
    )


def compute_outer_black_ratio(
    image_path: Path,
    edge_width_percent: float,
    black_pixel_threshold: int,
) -> float:
    """计算图像外环区域黑色像素占比。"""
    with Image.open(image_path) as img:
        gray = np.array(img.convert("L"), dtype=np.uint8)

    height, width = gray.shape
    if width < 4 or height < 4:
        return 0.0

    edge_ratio = edge_width_percent / 100.0
    edge_ratio = max(0.0, min(0.49, edge_ratio))
    margin_x = int(width * edge_ratio)
    margin_y = int(height * edge_ratio)

    if margin_x <= 0 or margin_y <= 0:
        return 0.0

    x1, x2 = margin_x, width - margin_x
    y1, y2 = margin_y, height - margin_y
    if x1 >= x2 or y1 >= y2:
        return 0.0

    outer_mask = np.ones((height, width), dtype=bool)
    outer_mask[y1:y2, x1:x2] = False
    outer_pixels = gray[outer_mask]
    if outer_pixels.size == 0:
        return 0.0

    black_count = int((outer_pixels <= black_pixel_threshold).sum())
    return black_count / float(outer_pixels.size)


def move_file(src: Path, dst: Path, dry_run: bool) -> None:
    """将文件移动到目标路径（自动创建父目录）。"""
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))


def process_split(
    split_dir: Path,
    scan_dir: Path,
    output_root: Path,
    timestamp: str,
    use_timestamp_suffix: bool,
    edge_width_percent: float,
    black_area_percent: float,
    black_pixel_threshold: int,
    dry_run: bool,
    debug: bool,
) -> SplitResult:
    """处理单个 split 目录，返回统计结果。"""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    has_labels_dir = labels_dir.is_dir()

    rel_split = split_dir.relative_to(scan_dir)
    out_split = output_root / rel_split
    suffix = f"_{timestamp}" if use_timestamp_suffix else ""
    out_images_rotated = out_split / f"images_rotated{suffix}"
    out_labels_rotated = out_split / f"labels_rotated{suffix}" if has_labels_dir else None

    image_paths = find_image_paths(images_dir)
    rotated_count = 0
    moved_json_count = 0
    moved_label_count = 0

    threshold_ratio = black_area_percent / 100.0

    print("=" * 70)
    print(f"处理 split : {split_dir}")
    print(f"  images   : {images_dir}")
    print(f"  labels   : {labels_dir if has_labels_dir else '(未找到)'}")
    print(f"  输出目录 : {out_split}")
    print(f"  图像总数 : {len(image_paths)}")
    print("=" * 70)

    for image_path in image_paths:
        ratio = compute_outer_black_ratio(
            image_path=image_path,
            edge_width_percent=edge_width_percent,
            black_pixel_threshold=black_pixel_threshold,
        )
        is_rotated = ratio >= threshold_ratio
        if debug:
            print(
                f"  [CHECK] {image_path.name} "
                f"outer_black_ratio={ratio:.4f} threshold={threshold_ratio:.4f} "
                f"=> {'ROTATED' if is_rotated else 'KEEP'}"
            )

        if not is_rotated:
            continue

        rotated_count += 1

        rel_image = image_path.relative_to(images_dir)
        dst_image = out_images_rotated / rel_image
        move_file(image_path, dst_image, dry_run=dry_run)

        json_path = image_path.with_suffix(".json")
        if json_path.is_file():
            dst_json = out_images_rotated / rel_image.with_suffix(".json")
            move_file(json_path, dst_json, dry_run=dry_run)
            moved_json_count += 1

        if has_labels_dir and out_labels_rotated is not None:
            label_path = labels_dir / rel_image.with_suffix(".txt")
            if label_path.is_file():
                dst_label = out_labels_rotated / rel_image.with_suffix(".txt")
                move_file(label_path, dst_label, dry_run=dry_run)
                moved_label_count += 1

    kept_count = len(image_paths) - rotated_count
    mode_prefix = "[DRY-RUN] " if dry_run else ""
    print(f"{mode_prefix}旋转图像数 : {rotated_count}")
    print(f"{mode_prefix}保留图像数 : {kept_count}")
    print(f"{mode_prefix}移动 JSON 数 : {moved_json_count}")
    print(f"{mode_prefix}移动 TXT 数  : {moved_label_count}")

    return SplitResult(
        split_dir=split_dir,
        out_images_rotated_dir=out_images_rotated,
        out_labels_rotated_dir=out_labels_rotated,
        total_images=len(image_paths),
        rotated_count=rotated_count,
        kept_count=kept_count,
        moved_json_count=moved_json_count,
        moved_label_count=moved_label_count,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    scan_dir = args.scan_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    use_timestamp_suffix = args.timestamp_suffix == "true"
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    if not scan_dir.is_dir():
        print(f"[错误] scan-dir 不存在：{scan_dir}", file=sys.stderr)
        return 1

    if args.edge_width_percent <= 0 or args.edge_width_percent >= 50:
        print("[错误] --edge-width-percent 必须在 (0, 50) 区间内", file=sys.stderr)
        return 1
    if args.black_area_percent < 0 or args.black_area_percent > 100:
        print("[错误] --black-area-percent 必须在 [0, 100] 区间内", file=sys.stderr)
        return 1
    if args.black_pixel_threshold < 0 or args.black_pixel_threshold > 255:
        print("[错误] --black-pixel-threshold 必须在 [0, 255] 区间内", file=sys.stderr)
        return 1

    print(f"扫描目录            : {scan_dir}")
    print(f"输出根目录          : {output_root}")
    print(f"外环宽度阈值(%)     : {args.edge_width_percent}")
    print(f"外环黑色占比阈值(%) : {args.black_area_percent}")
    print(f"黑色像素阈值        : <= {args.black_pixel_threshold}")
    print(f"时间戳后缀          : {use_timestamp_suffix}")
    print(f"预览模式            : {args.dry_run}")

    split_dirs = scan_split_dirs(scan_dir)
    if not split_dirs:
        print(f"[错误] 在 {scan_dir} 下未找到包含 images/ 子目录的目录。", file=sys.stderr)
        return 1

    print(f"\n找到 {len(split_dirs)} 个 split 目录：")
    for split_dir in split_dirs:
        print(f"  {split_dir}")

    results: List[SplitResult] = []
    for split_dir in split_dirs:
        result = process_split(
            split_dir=split_dir,
            scan_dir=scan_dir,
            output_root=output_root,
            timestamp=timestamp,
            use_timestamp_suffix=use_timestamp_suffix,
            edge_width_percent=args.edge_width_percent,
            black_area_percent=args.black_area_percent,
            black_pixel_threshold=args.black_pixel_threshold,
            dry_run=args.dry_run,
            debug=args.debug,
        )
        results.append(result)

    total_images = sum(r.total_images for r in results)
    total_rotated = sum(r.rotated_count for r in results)
    total_kept = sum(r.kept_count for r in results)
    total_json = sum(r.moved_json_count for r in results)
    total_txt = sum(r.moved_label_count for r in results)

    print(f"\n{'=' * 70}")
    print("全部完成，汇总：")
    print(f"  处理 split 数 : {len(results)}")
    print(f"  输入图像总数  : {total_images}")
    print(f"  旋转图像总数  : {total_rotated}")
    print(f"  保留图像总数  : {total_kept}")
    print(f"  移动 JSON 总数 : {total_json}")
    print(f"  移动 TXT 总数  : {total_txt}")
    print(f"{'=' * 70}")

    if args.print_output_dir:
        for result in results:
            print(f"OUTPUT_DIR:{result.split_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
