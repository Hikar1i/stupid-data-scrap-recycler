#!/usr/bin/env python3
"""将 YOLO 标签文件中的类别 ID 原地或输出到新目录进行重映射。

映射格式：<源ID>:<目标ID>[,<源ID>:<目标ID>...]  例如：8:0,11:1
"""
import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 YOLO 标签文件中的类别 ID 进行重映射，支持原地修改或输出到新目录。"
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
    parser.add_argument(
        "--mapping",
        type=str,
        required=True,
        help="类别 ID 映射关系，例如 '8:0,11:1'。",
    )
    parser.add_argument(
        "--inplace",
        choices=["true", "false"],
        default="false",
        help="为 true 时直接修改原始标签文件，--output-dir 参数将被忽略。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "重映射结果输出目录，--inplace=true 时无效。"
            "未指定时在每个 labels/ 同级目录创建 labels_remapping_<时间戳>/。"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式，不修改或创建任何文件。",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="打印每个文件、每行的替换/跳过详情。",
    )
    parser.add_argument(
        "--print-output-dir",
        action="store_true",
        help="在末尾打印 OUTPUT_DIR:<路径> 行，供 pipeline wrapper 捕获。",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 解析工具函数
# ---------------------------------------------------------------------------

def parse_mapping(raw: str) -> Dict[int, int]:
    """解析映射字符串 '8:0,11:1' → {8: 0, 11: 1}"""
    mapping: Dict[int, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+):(\d+)", part)
        if not m:
            raise ValueError(f"映射格式不合法（期望 '整数:整数'）：'{part}'")
        src, dst = int(m.group(1)), int(m.group(2))
        if src in mapping:
            raise ValueError(f"映射中存在重复的源 ID：{src}")
        mapping[src] = dst
    if not mapping:
        raise ValueError("映射为空。")
    return mapping


def parse_label_class_id(line: str) -> Optional[int]:
    stripped = line.strip()
    if not stripped:
        return None
    first = stripped.split()[0]
    return int(first) if first.isdigit() else None


# ---------------------------------------------------------------------------
# 数据集目录结构探测
# ---------------------------------------------------------------------------

def find_labels_dirs(source_dir: Path) -> List[Path]:
    """递归查找 source_dir 下的所有 labels 子目录。

    若 rglob("labels") 未找到任何结果，则检查 source_dir 自身是否包含 .txt
    文件——若是，则将其视为 labels 目录直接返回。此回退逻辑支持 pipeline 将
    labels_deduped_xxx/ 等非标准命名的标注目录直接作为 --source-dir 传入。
    """
    found = sorted([p for p in source_dir.rglob("labels") if p.is_dir()])
    if not found:
        if any(source_dir.glob("*.txt")):
            return [source_dir]
        raise FileNotFoundError(f"在 {source_dir} 下未找到任何 labels 目录")
    return found


# ---------------------------------------------------------------------------
# 冲突检测与重映射
# ---------------------------------------------------------------------------

def remap_lines(
    lines: List[str],
    mapping: Dict[int, int],
    label_path: Path,
    debug: bool,
) -> Tuple[List[str], int, int, int]:
    """
    对标签文件的每一行应用 ID 映射。

    返回 (新行列表, 已替换行数, 冲突跳过行数, 未变更行数)。
    若目标 ID 已存在于文件中其他不属于同一映射对的行，则跳过该行替换以避免 ID 重复。
    """
    # 收集文件中已存在的所有类别 ID
    all_ids: Set[int] = set()
    for line in lines:
        cid = parse_label_class_id(line)
        if cid is not None:
            all_ids.add(cid)

    # 确定哪些目标 ID 存在冲突：目标 ID 已存在于非源 ID 的其他行中
    conflict_targets: Set[int] = set()
    for src, dst in mapping.items():
        if dst in all_ids and dst != src:
            # dst 存在于文件中非 src 行 → 冲突
            non_src_ids = all_ids - {src}
            if dst in non_src_ids:
                conflict_targets.add(dst)

    new_lines: List[str] = []
    replaced = 0
    skipped = 0
    unchanged = 0

    for line in lines:
        cid = parse_label_class_id(line)
        if cid is None:
            new_lines.append(line)
            continue

        if cid not in mapping:
            new_lines.append(line)
            unchanged += 1
            continue

        dst = mapping[cid]
        if dst in conflict_targets:
            # 目标 ID 冲突，保留原行不替换
            new_lines.append(line)
            skipped += 1
            if debug:
                print(
                    f"  [冲突跳过] {label_path}: "
                    f"行 '{line.strip()}' → 目标 ID {dst} 已存在于文件中"
                )
        else:
            rest = line.strip()[len(str(cid)):]
            new_line = f"{dst}{rest}"
            new_lines.append(new_line)
            replaced += 1
            if debug:
                print(
                    f"  [REMAP] {label_path}: "
                    f"'{line.strip()}' → '{new_line}'"
                )

    return new_lines, replaced, skipped, unchanged


# ---------------------------------------------------------------------------
# 输出路径解析
# ---------------------------------------------------------------------------

def output_labels_dir(
    labels_dir: Path,
    inplace: bool,
    output_dir: Optional[Path],
    timestamp: str,
) -> Path:
    if inplace:
        return labels_dir
    if output_dir is not None:
        # 在 output_dir 内部镜像相对目录结构
        try:
            rel = labels_dir.parent.relative_to(labels_dir.parent.parent.parent)
        except ValueError:
            rel = Path(labels_dir.parent.name)
        return output_dir / rel / f"labels_remapping_{timestamp}"
    # 默认：在原始 labels/ 同级目录下创建输出目录
    return labels_dir.parent / f"labels_remapping_{timestamp}"


# ---------------------------------------------------------------------------
# 核心处理逻辑
# ---------------------------------------------------------------------------

def process_labels_dir(
    labels_dir: Path,
    mapping: Dict[int, int],
    out_dir: Path,
    inplace: bool,
    dry_run: bool,
    debug: bool,
    timestamp: str,
) -> Dict[str, int]:
    stats: Dict[str, int] = defaultdict(int)
    label_files = sorted(labels_dir.glob("*.txt"))
    stats["total_files"] = len(label_files)

    if not label_files:
        return stats

    if not dry_run and not inplace:
        out_dir.mkdir(parents=True, exist_ok=True)

    for lf in label_files:
        raw = lf.read_text(encoding="utf-8")
        lines = raw.splitlines()
        new_lines, replaced, skipped, unchanged = remap_lines(lines, mapping, lf, debug)

        if skipped > 0:
            stats["files_with_conflicts"] += 1
            stats["total_conflict_lines"] += skipped

        if replaced > 0:
            stats["files_modified"] += 1
            stats["total_replaced_lines"] += replaced

        if not dry_run:
            dst_path = out_dir / lf.name if not inplace else lf
            dst_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")

    return stats


# ---------------------------------------------------------------------------
# 入口函数
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if not args.source_dir.is_dir():
        print(f"错误：source-dir 不存在：{args.source_dir}", file=sys.stderr)
        return 1

    try:
        mapping = parse_mapping(args.mapping)
    except ValueError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1

    inplace = args.inplace == "true"
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    try:
        labels_dirs = find_labels_dirs(args.source_dir)
    except FileNotFoundError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1

    print("=" * 90)
    print("YOLO 标签序号重映射")
    print("=" * 90)
    print(f"source_dir:  {args.source_dir}")
    print(f"mapping:     {mapping}")
    print(f"inplace:     {inplace}")
    print(f"dry_run:     {args.dry_run}")
    print(f"labels_dirs_found: {len(labels_dirs)}")
    for ld in labels_dirs:
        print(f"  {ld}")
    print("-" * 90)

    all_output_dirs: List[Path] = []
    total_conflict_files = 0
    total_conflict_lines = 0

    for labels_dir in labels_dirs:
        out_dir = output_labels_dir(labels_dir, inplace, args.output_dir, timestamp)
        all_output_dirs.append(out_dir)

        stats = process_labels_dir(
            labels_dir=labels_dir,
            mapping=mapping,
            out_dir=out_dir,
            inplace=inplace,
            dry_run=args.dry_run,
            debug=args.debug,
            timestamp=timestamp,
        )

        total_conflict_files += stats["files_with_conflicts"]
        total_conflict_lines += stats["total_conflict_lines"]

        mode_tag = "[DRY-RUN]" if args.dry_run else ("[INPLACE]" if inplace else "[OUTPUT]")
        print(f"{mode_tag} labels_dir: {labels_dir}")
        print(f"  output_dir:     {out_dir}")
        print(f"  total_files:    {stats['total_files']}")
        print(f"  files_modified: {stats['files_modified']}")
        print(f"  replaced_lines: {stats['total_replaced_lines']}")
        print(f"  conflict_files: {stats['files_with_conflicts']}")
        print(f"  conflict_lines: {stats['total_conflict_lines']}")
        print("-" * 90)

    if total_conflict_files > 0 or total_conflict_lines > 0:
        print("!" * 90)
        print("警告：检测到冲突 — 部分行未完成重映射！")
        print(
            f"  共 {total_conflict_files} 个文件存在冲突行，"
            f"跳过了 {total_conflict_lines} 行。"
        )
        print(
            "  原因：目标类别 ID 已存在于同一标签文件的其他行中。\n"
            "  当前输出标签为【部分重映射】状态，结果可能不正确。\n"
            "  解决方法：检查并修正 --mapping 参数后重新运行。"
        )
        print("!" * 90)

    if args.print_output_dir:
        for out_dir in all_output_dirs:
            print(f"OUTPUT_DIR:{out_dir}")

    print("=" * 90)
    return 0


if __name__ == "__main__":
    sys.exit(main())
