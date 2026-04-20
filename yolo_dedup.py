#!/usr/bin/env python3
"""
对 YOLO 数据集各 split 目录（train/valid/test）执行去重去相似操作。

扫描 --scan-dir 下所有包含 images/ 子目录的目录，对每个 split 独立执行
多重验证去重（ResNet50 深度特征 + dHash + SSIM），结果以硬链接方式输出到
--output-root，保留原始目录层级结构。

运行环境：/home/ieds/.conda/envs/cu12-dev/bin/python
"""

import argparse
import csv
import hashlib
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms


# ── 常量 ─────────────────────────────────────────────────────────────────────

IMG_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class DupGroup:
    """单个重复图像组的统计信息。"""
    group_id: int
    paths: List[Path]
    method: str
    resnet50_sim: float
    dhash_sim: Optional[float]
    ssim: Optional[float]


@dataclass
class DedupeResult:
    """单个 split 去重操作的结果摘要。"""
    split_dir: Path
    total_images: int
    unique_count: int
    dup_group_count: int
    out_images_dir: Path
    out_labels_dir: Optional[Path]
    out_dups_dir: Path
    report_path: Path
    dup_groups: List[DupGroup] = field(default_factory=list)


# ── 图像工具 ──────────────────────────────────────────────────────────────────

def iter_images(folder: Path) -> List[Path]:
    """返回 folder 下所有图像文件（递归，按路径排序）。"""
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def file_md5(path: Path, chunk_size: int = 8192) -> str:
    """计算文件 MD5 哈希值。"""
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def safe_open_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def dhash64(img: Image.Image, size: Tuple[int, int] = (9, 8)) -> int:
    """计算 64 位差分哈希（dHash）。"""
    g = img.convert("L").resize(size, Image.Resampling.BILINEAR)
    px = list(g.getdata())
    w, h = size
    bits = 0
    bitpos = 0
    for y in range(h):
        row = px[y * w: (y + 1) * w]
        for x in range(w - 1):
            if row[x] > row[x + 1]:
                bits |= 1 << bitpos
            bitpos += 1
    return bits


def dhash_similarity(a: int, b: int) -> float:
    """计算两个 dHash 之间的相似度（1 - 汉明距离/64）。"""
    return 1.0 - (a ^ b).bit_count() / 64.0


def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """计算两张图像的全局结构相似度（SSIM）。"""
    size = (256, 256)
    a = np.array(img1.convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float64)
    b = np.array(img2.convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float64)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu1, mu2 = a.mean(), b.mean()
    s1 = ((a - mu1) ** 2).mean()
    s2 = ((b - mu2) ** 2).mean()
    s12 = ((a - mu1) * (b - mu2)).mean()
    return float(
        ((2 * mu1 * mu2 + C1) * (2 * s12 + C2))
        / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
    )


# ── ResNet50 特征提取 ─────────────────────────────────────────────────────────

def build_feature_extractor(device: torch.device) -> torch.nn.Module:
    """加载 ResNet50（去掉分类头，输出 2048 维特征向量）。"""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model


def default_preprocess() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_features(
    paths: List[Path],
    model: torch.nn.Module,
    preprocess: transforms.Compose,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """批量提取图像特征，返回 L2 归一化后的特征矩阵（shape: [N, 2048]）。"""
    all_feats: List[torch.Tensor] = []
    for i in tqdm(range(0, len(paths), batch_size), desc="特征提取", unit="batch"):
        batch: List[torch.Tensor] = []
        for p in paths[i: i + batch_size]:
            try:
                batch.append(preprocess(safe_open_image(p)))
            except Exception:
                batch.append(torch.zeros(3, 224, 224))
        with torch.no_grad():
            feats = model(torch.stack(batch).to(device))
        all_feats.append(feats.cpu())
    feats = torch.cat(all_feats, dim=0)
    return F.normalize(feats, p=2, dim=1)


# ── 标签与标注文件查找 ────────────────────────────────────────────────────────

def resolve_labels_dir(images_dir: Path) -> Optional[Path]:
    """从 images/ 目录推断对应的 labels/ 目录。
    优先尝试将路径中的 'images' 替换为 'labels'，其次尝试同级 labels/ 目录。
    """
    parts = list(images_dir.parts)
    for i, part in enumerate(parts):
        if part == "images":
            cand = Path(*parts[:i], "labels", *parts[i + 1:])
            if cand.is_dir():
                return cand
    sibling = images_dir.parent / "labels"
    if sibling.is_dir():
        return sibling
    return None


def find_label_file(image_path: Path, labels_dir: Path) -> Optional[Path]:
    """查找图像对应的 YOLO .txt 标注文件。"""
    cand = labels_dir / (image_path.stem + ".txt")
    return cand if cand.is_file() else None


def find_json_file(image_path: Path) -> Optional[Path]:
    """查找图像同目录下同名的 labelme .json 标注文件。"""
    cand = image_path.with_suffix(".json")
    return cand if cand.is_file() else None


# ── 硬链接工具 ────────────────────────────────────────────────────────────────

def hardlink(src: Path, dst: Path) -> None:
    """在 dst 创建指向 src 的硬链接。
    若目标已存在则跳过；跨文件系统时自动降级为文件复制并打印警告。
    """
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError as e:
        print(
            f"[警告] 硬链接失败（{e}），已降级为文件复制：{src} → {dst}",
            file=sys.stderr,
        )
        shutil.copy2(str(src), str(dst))


def link_to_deduped(
    image_path: Path,
    labels_dir: Optional[Path],
    dst_images_dir: Path,
    dst_labels_dir: Optional[Path],
    warn_context: str,
) -> None:
    """将图像及其关联标注文件（.json / .txt）硬链接到去重保留目录。"""
    hardlink(image_path, dst_images_dir / image_path.name)

    json_src = find_json_file(image_path)
    if json_src is not None:
        hardlink(json_src, dst_images_dir / json_src.name)

    if labels_dir is not None and dst_labels_dir is not None:
        txt_src = find_label_file(image_path, labels_dir)
        if txt_src is not None:
            hardlink(txt_src, dst_labels_dir / txt_src.name)
        else:
            print(
                f"[警告][{warn_context}] 未找到 YOLO 标注文件：{image_path.name}",
                file=sys.stderr,
            )


def link_to_group(
    image_path: Path,
    labels_dir: Optional[Path],
    group_dir: Path,
) -> None:
    """将图像及其关联标注文件（.json / .txt）硬链接到 duplicates/group_XXXXXX 目录。"""
    hardlink(image_path, group_dir / image_path.name)

    json_src = find_json_file(image_path)
    if json_src is not None:
        hardlink(json_src, group_dir / json_src.name)

    if labels_dir is not None:
        txt_src = find_label_file(image_path, labels_dir)
        if txt_src is not None:
            hardlink(txt_src, group_dir / txt_src.name)


# ── 目录扫描 ─────────────────────────────────────────────────────────────────

def scan_split_dirs(scan_dir: Path) -> List[Path]:
    """扫描 scan_dir 下所有直接包含 images/ 子目录的目录（即 split 目录）。"""
    splits: List[Path] = []
    for images_subdir in sorted(scan_dir.rglob("images")):
        if images_subdir.is_dir() and images_subdir.parent != scan_dir:
            splits.append(images_subdir.parent)
        elif images_subdir.is_dir() and images_subdir.parent == scan_dir:
            # scan_dir 本身就是 split 目录（直接含 images/）
            splits.append(images_subdir.parent)
    # 去重（防止 rglob 重复）
    seen: Set[Path] = set()
    result: List[Path] = []
    for p in splits:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


# ── 核心去重逻辑 ─────────────────────────────────────────────────────────────

def _compute_group_metrics(
    indices: List[int],
    feats: torch.Tensor,
    dhashes: List[Optional[int]],
    all_paths: List[Path],
    md5_hashes: List[Optional[str]],
) -> Tuple[str, float, Optional[float], Optional[float]]:
    """计算重复组的相似度统计指标，供报告使用。"""
    md5s = {md5_hashes[i] for i in indices if md5_hashes[i]}
    if len(md5s) == 1:
        return "md5", 1.0, None, None

    feat_sims: List[float] = []
    dh_sims: List[float] = []
    ssim_vals: List[float] = []
    for a in range(len(indices)):
        for b in range(a + 1, len(indices)):
            ia, ib = indices[a], indices[b]
            feat_sims.append(torch.dot(feats[ia], feats[ib]).item())
            if dhashes[ia] is not None and dhashes[ib] is not None:
                dh_sims.append(dhash_similarity(dhashes[ia], dhashes[ib]))
            try:
                ssim_vals.append(
                    compute_ssim(safe_open_image(all_paths[ia]), safe_open_image(all_paths[ib]))
                )
            except Exception:
                pass

    r50 = sum(feat_sims) / len(feat_sims) if feat_sims else 1.0
    dh = sum(dh_sims) / len(dh_sims) if dh_sims else None
    ssim = sum(ssim_vals) / len(ssim_vals) if ssim_vals else None
    return "multi_verify", r50, dh, ssim


def dedup_split(
    split_dir: Path,
    scan_dir: Path,
    output_root: Path,
    timestamp: str,
    use_timestamp_suffix: bool,
    threshold: float,
    dhash_threshold: float,
    ssim_threshold: float,
    model: torch.nn.Module,
    preprocess: transforms.Compose,
    batch_size: int,
    device: torch.device,
) -> Optional[DedupeResult]:
    """对单个 split 目录的 images/ 执行去重，以硬链接方式输出结果。"""
    sfx = f"_{timestamp}" if use_timestamp_suffix else ""
    images_dir = split_dir / "images"
    labels_dir = resolve_labels_dir(images_dir)

    rel = split_dir.relative_to(scan_dir)
    out_split = output_root / rel
    out_images = out_split / f"images_deduped{sfx}"
    out_labels = out_split / f"labels_deduped{sfx}"
    out_dups = out_split / f"duplicates{sfx}"
    report_path = out_split / f"duplicate_report{sfx}.csv"

    all_paths = iter_images(images_dir)
    n = len(all_paths)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"处理 split : {split_dir}")
    print(f"  images   : {images_dir}")
    print(f"  labels   : {labels_dir or '(未找到，标注文件不会被处理)'}")
    print(f"  图像总数 : {n}")
    print(f"  输出目录 : {out_split}")
    print(sep)

    if n == 0:
        print("[警告] images/ 目录下未找到图像，跳过该 split。", file=sys.stderr)
        return None

    # Step 1: MD5 精确哈希
    print("\nStep 1 / 4：计算 MD5...")
    md5_hashes: List[Optional[str]] = [None] * n
    for i, p in enumerate(tqdm(all_paths, desc="MD5", unit="img")):
        try:
            md5_hashes[i] = file_md5(p)
        except Exception:
            pass

    # Step 2: dHash 感知哈希
    print("\nStep 2 / 4：计算 dHash...")
    dhashes: List[Optional[int]] = [None] * n
    for i, p in enumerate(tqdm(all_paths, desc="dHash", unit="img")):
        try:
            dhashes[i] = dhash64(safe_open_image(p))
        except Exception:
            pass

    # Step 3: ResNet50 深度特征
    print("\nStep 3 / 4：提取 ResNet50 特征...")
    feats = extract_features(all_paths, model, preprocess, device, batch_size)

    # Step 4: 检测重复（MD5 精确匹配 → ResNet50 + dHash + SSIM 多重验证）
    print("\nStep 4 / 4：检测重复图像...")
    processed: Set[int] = set()
    dup_groups: List[List[int]] = []
    dup_group_map: Dict[int, int] = {}
    unique_indices: List[int] = []

    for i in tqdm(range(n), desc="检测重复", unit="img"):
        if i in processed:
            continue

        is_dup = False
        matched_gid = -1

        for j in range(i):
            if j not in processed:
                continue

            # MD5 精确匹配：直接判定为重复
            if md5_hashes[i] and md5_hashes[j] and md5_hashes[i] == md5_hashes[j]:
                is_dup = True
            else:
                # ResNet50 粗筛
                if torch.dot(feats[i], feats[j]).item() < threshold:
                    continue
                # dHash 中筛
                if dhashes[i] is not None and dhashes[j] is not None:
                    if dhash_similarity(dhashes[i], dhashes[j]) < dhash_threshold:
                        continue
                # SSIM 精筛
                try:
                    ssim_val = compute_ssim(
                        safe_open_image(all_paths[i]),
                        safe_open_image(all_paths[j]),
                    )
                    if ssim_val < ssim_threshold:
                        continue
                except Exception:
                    continue
                is_dup = True

            if is_dup:
                if j in dup_group_map:
                    matched_gid = dup_group_map[j]
                else:
                    matched_gid = len(dup_groups)
                    dup_groups.append([j])
                    dup_group_map[j] = matched_gid
                break

        if is_dup:
            dup_groups[matched_gid].append(i)
            dup_group_map[i] = matched_gid
        else:
            unique_indices.append(i)

        processed.add(i)

    total_dup_imgs = sum(len(g) for g in dup_groups)
    truly_unique = len(unique_indices) - len(dup_groups)
    print("\n去重统计：")
    print(f"  输入图像总数       : {n}")
    print(f"  完全唯一图像       : {truly_unique}（无任何重复）")
    print(f"  重复组数           : {len(dup_groups)}")
    print(f"  重复组内图像总数   : {total_dup_imgs}（含每组代表图）")
    print(f"  去重后保留图像数   : {len(unique_indices)}")

    # Step 5: 创建输出目录并建立硬链接
    print("\nStep 5：创建硬链接...")
    out_images.mkdir(parents=True, exist_ok=True)
    if labels_dir is not None:
        out_labels.mkdir(parents=True, exist_ok=True)
    out_dups.mkdir(parents=True, exist_ok=True)

    # 去重保留图像（unique_indices = 真唯一图像 + 每个重复组的代表图）
    for idx in tqdm(unique_indices, desc="链接保留图像", unit="img"):
        link_to_deduped(
            image_path=all_paths[idx],
            labels_dir=labels_dir,
            dst_images_dir=out_images,
            dst_labels_dir=out_labels if labels_dir is not None else None,
            warn_context=f"deduped/{rel}",
        )

    # 重复组（代表图 + 所有重复图均进入 duplicates/group_XXXXXX）
    dup_group_records: List[DupGroup] = []
    for gid, indices in enumerate(
        tqdm(dup_groups, desc="链接重复组", unit="group"), start=1
    ):
        group_dir = out_dups / f"group_{gid:06d}"
        group_dir.mkdir(parents=True, exist_ok=True)
        group_paths = [all_paths[i] for i in indices]

        for img_path in group_paths:
            link_to_group(img_path, labels_dir, group_dir)

        method, r50, dh, ssim = _compute_group_metrics(
            indices, feats, dhashes, all_paths, md5_hashes
        )
        dup_group_records.append(DupGroup(gid, group_paths, method, r50, dh, ssim))

    # Step 6: 输出 CSV 报告
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "group_id", "image_count", "method",
            "resnet50_sim", "dhash_sim", "ssim", "image_paths",
        ])
        for g in dup_group_records:
            w.writerow([
                g.group_id, len(g.paths), g.method,
                f"{g.resnet50_sim:.6f}",
                f"{g.dhash_sim:.6f}" if g.dhash_sim is not None else "",
                f"{g.ssim:.6f}" if g.ssim is not None else "",
                " | ".join(str(p) for p in g.paths),
            ])

    print(f"\n{sep}")
    print(f"完成：{rel}")
    print(f"  images_deduped : {out_images}")
    if labels_dir is not None:
        print(f"  labels_deduped : {out_labels}")
    print(f"  duplicates     : {out_dups}")
    print(f"  报告           : {report_path}")
    print(sep)

    return DedupeResult(
        split_dir=split_dir,
        total_images=n,
        unique_count=len(unique_indices),
        dup_group_count=len(dup_groups),
        out_images_dir=out_images,
        out_labels_dir=out_labels if labels_dir is not None else None,
        out_dups_dir=out_dups,
        report_path=report_path,
        dup_groups=dup_group_records,
    )


# ── 参数解析 ─────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "对 YOLO 数据集各 split 目录执行去重去相似操作。\n"
            "扫描 --scan-dir 下所有包含 images/ 子目录的目录，逐 split 独立处理，\n"
            "结果以硬链接方式保存至 --output-root（保留原目录层级）。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--scan-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="扫描根目录，递归查找所有包含 images/ 子目录的 split 目录",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        required=False,
        default=None,
        metavar="DIR",
        help=(
            "输出根目录，保留 scan-dir 下的原始目录层级。"
            "未指定时默认使用 --scan-dir 本身，"
            "即将 images_deduped*/labels_deduped*/duplicates* 写入各 split 目录内（推荐）。"
        ),
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        metavar="FLOAT",
        help="ResNet50 深度特征余弦相似度阈值（默认 0.85）",
    )
    p.add_argument(
        "--dhash-threshold",
        type=float,
        default=0.90,
        metavar="FLOAT",
        help="dHash 感知哈希相似度阈值（默认 0.90）",
    )
    p.add_argument(
        "--ssim-threshold",
        type=float,
        default=0.85,
        metavar="FLOAT",
        help="SSIM 结构相似度阈值（默认 0.85）",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="特征提取 batch 大小（默认 32）",
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="计算设备：cuda / cpu（默认自动检测）",
    )
    p.add_argument(
        "--timestamp-suffix",
        choices=["true", "false"],
        default="true",
        help=(
            "输出目录名是否带时间戳后缀（默认 true）。\n"
            "false 时多次运行将覆盖上次结果。"
        ),
    )
    p.add_argument(
        "--print-output-dir",
        action="store_true",
        help=(
            "每个 split 处理完成后逐行打印 OUTPUT_DIR:<labels_deduped_dir>，"
            "供 pipeline wrapper（reindex / cvtlabelme 阶段）捕获。"
        ),
    )
    return p.parse_args(argv)


# ── 主函数 ───────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    scan_dir = args.scan_dir.expanduser().resolve()
    # output_root 未指定时默认与 scan_dir 相同，结果写入各 split 子目录内
    output_root = (
        args.output_root.expanduser().resolve() if args.output_root is not None else scan_dir
    )
    use_ts = args.timestamp_suffix == "true"
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    if not scan_dir.is_dir():
        print(f"[错误] scan-dir 不存在：{scan_dir}", file=sys.stderr)
        return 1

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"扫描目录   : {scan_dir}")
    print(f"输出根目录 : {output_root}")
    print(f"计算设备   : {device}")
    print(f"时间戳后缀 : {use_ts}")
    print(f"ResNet50 阈值 : {args.threshold}")
    print(f"dHash 阈值    : {args.dhash_threshold}")
    print(f"SSIM 阈值     : {args.ssim_threshold}")
    print("多重验证   : 始终开启（MD5 + ResNet50 + dHash + SSIM）")

    split_dirs = scan_split_dirs(scan_dir)
    if not split_dirs:
        print(f"[错误] 在 {scan_dir} 下未找到包含 images/ 子目录的目录。", file=sys.stderr)
        return 1

    print(f"\n找到 {len(split_dirs)} 个 split 目录：")
    for d in split_dirs:
        print(f"  {d}")

    # 仅加载一次模型，所有 split 共享
    print("\n加载 ResNet50 特征提取器...")
    model = build_feature_extractor(device)
    preprocess = default_preprocess()

    results: List[DedupeResult] = []
    for split_dir in split_dirs:
        result = dedup_split(
            split_dir=split_dir,
            scan_dir=scan_dir,
            output_root=output_root,
            timestamp=timestamp,
            use_timestamp_suffix=use_ts,
            threshold=args.threshold,
            dhash_threshold=args.dhash_threshold,
            ssim_threshold=args.ssim_threshold,
            model=model,
            preprocess=preprocess,
            batch_size=args.batch_size,
            device=device,
        )
        if result is not None:
            results.append(result)

    # 全局汇总
    if results:
        print(f"\n{'='*70}")
        print("全部完成，汇总：")
        total_in = sum(r.total_images for r in results)
        total_kept = sum(r.unique_count for r in results)
        total_groups = sum(r.dup_group_count for r in results)
        print(f"  处理 split 数   : {len(results)}")
        print(f"  输入图像总数    : {total_in}")
        print(f"  去重后保留图像  : {total_kept}")
        print(f"  重复组总数      : {total_groups}")
        print(f"{'='*70}")

    # pipeline 模式：逐 split 输出 labels_deduped 目录，供 reindex / cvtlabelme 捕获
    if args.print_output_dir:
        for result in results:
            if result.out_labels_dir is not None:
                print(f"OUTPUT_DIR:{result.out_labels_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
