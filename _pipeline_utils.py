"""流水线 wrapper 脚本共享工具模块。

管道阶段：filter → dedup → reindex → cvtlabelme
每个阶段从上一阶段的 stdout 中解析 OUTPUT_DIR:<路径> 行，
作为下一阶段的源目录传入。
"""
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


OUTPUT_DIR_PREFIX = "OUTPUT_DIR:"


def build_dedup_command(
    script_path: Path,
    scan_dir: Path,
    dedup_cfg: dict,
    print_output_dir: bool,
) -> List[str]:
    """构建 dedup_yolo_dataset.py 命令。

    dedup_cfg 支持的键：
      python          (可选) 专用 Python 解释器路径，默认用 sys.executable
      output_root     (可选) 去重结果输出根目录；未配置时使用 scan_dir（推荐）
      threshold       (可选) ResNet50 相似度阈值
      dhash_threshold (可选) dHash 相似度阈值
      ssim_threshold  (可选) SSIM 阈值
      batch_size      (可选) 特征提取 batch 大小
      device          (可选) 计算设备 cuda/cpu
      timestamp_suffix(可选) 输出目录名是否带时间戳后缀
    """
    # output_root 未配置时默认等于 scan_dir（dedup 结果写入 filter 输出目录内）
    output_root = dedup_cfg.get("output_root") or str(scan_dir)

    python_bin = dedup_cfg.get("python") or sys.executable
    cmd: List[str] = [
        python_bin, str(script_path),
        "--scan-dir", str(scan_dir),
        "--output-root", str(output_root),
    ]

    for key, arg in [
        ("threshold", "--threshold"),
        ("dhash_threshold", "--dhash-threshold"),
        ("ssim_threshold", "--ssim-threshold"),
        ("batch_size", "--batch-size"),
        ("device", "--device"),
    ]:
        val = dedup_cfg.get(key)
        if val is not None:
            cmd.extend([arg, str(val)])

    ts = dedup_cfg.get("timestamp_suffix")
    if ts is not None:
        cmd.extend(["--timestamp-suffix", str(ts).lower()])

    if print_output_dir:
        cmd.append("--print-output-dir")

    return cmd


def run_stage(cmd: List[str], stage_name: str, print_command: bool) -> List[Path]:
    """执行一个管道阶段的子进程命令。

    实时流式输出到终端，同时解析 OUTPUT_DIR:<路径> 行。
    返回捕获到的输出目录列表，子进程非零返回值时抛出 SystemExit。
    """
    if print_command:
        print(f"[pipeline] 执行阶段 '{stage_name}':")
        print("  " + " ".join(str(c) for c in cmd))

    output_dirs: List[Path] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        stripped = line.rstrip("\n")
        if stripped.startswith(OUTPUT_DIR_PREFIX):
            raw = stripped[len(OUTPUT_DIR_PREFIX):].strip()
            if raw:
                output_dirs.append(Path(raw))
    proc.wait()

    if proc.returncode != 0:
        print(
            f"[pipeline] 阶段 '{stage_name}' 失败，退出码：{proc.returncode}",
            file=sys.stderr,
        )
        sys.exit(proc.returncode)

    return output_dirs


def build_reindex_commands(
    script_path: Path,
    source_dirs: List[Path],
    reindex_cfg: dict,
    print_output_dir: bool,
) -> List[List[str]]:
    """为每个源目录构建 remap_yolo_labels.py 命令列表。"""
    mapping = reindex_cfg.get("mapping", "")
    if not mapping:
        print("[pipeline] 错误：reindex 配置缺少 'mapping' 字段", file=sys.stderr)
        sys.exit(1)

    inplace = str(reindex_cfg.get("inplace", "false")).lower()
    dry_run = reindex_cfg.get("dry_run", False)
    debug = reindex_cfg.get("debug", False)
    output_dir = reindex_cfg.get("output_dir")

    cmds = []
    for src in source_dirs:
        cmd = [sys.executable, str(script_path), "--source-dir", str(src), "--mapping", mapping]
        cmd.extend(["--inplace", inplace])
        if output_dir:
            cmd.extend(["--output-dir", str(output_dir)])
        if dry_run:
            cmd.append("--dry-run")
        if debug:
            cmd.append("--debug")
        if print_output_dir:
            cmd.append("--print-output-dir")
        cmds.append(cmd)
    return cmds


def build_cvtlabelme_commands(
    script_path: Path,
    source_dirs: List[Path],
    cvtlabelme_cfg: dict,
    print_output_dir: bool,
) -> List[List[str]]:
    """为每个源目录构建 yolo_to_labelme.py 命令列表。"""
    mapping = cvtlabelme_cfg.get("mapping", "")
    classes_file = cvtlabelme_cfg.get("classes_file")

    if not mapping and not classes_file:
        print(
            "[pipeline] 错误：cvtlabelme 配置必须有 'mapping' 或 'classes_file' 字段",
            file=sys.stderr,
        )
        sys.exit(1)

    overwrite = str(cvtlabelme_cfg.get("overwrite", "false")).lower()
    dry_run = cvtlabelme_cfg.get("dry_run", False)
    debug = cvtlabelme_cfg.get("debug", False)
    output_dir = cvtlabelme_cfg.get("output_dir")

    cmds = []
    for src in source_dirs:
        cmd = [sys.executable, str(script_path), "--source-dir", str(src)]
        if mapping:
            cmd.extend(["--mapping", mapping])
        elif classes_file:
            cmd.extend(["--classes-file", str(classes_file)])
        cmd.extend(["--overwrite", overwrite])
        if output_dir:
            cmd.extend(["--output-dir", str(output_dir)])
        if dry_run:
            cmd.append("--dry-run")
        if debug:
            cmd.append("--debug")
        if print_output_dir:
            cmd.append("--print-output-dir")
        cmds.append(cmd)
    return cmds


def run_pipeline_stages(
    filter_output_dirs: List[Path],
    script_root: Path,
    reindex_cfg: Optional[dict],
    cvtlabelme_cfg: Optional[dict],
    print_command: bool,
    dedup_cfg: Optional[dict] = None,
) -> None:
    """在 filter 阶段完成后，依次调度 dedup、reindex 和或 cvtlabelme 阶段。

    filter_output_dirs：filter 阶段产生的输出目录列表（通常为含各 split 的 merge 根目录）。
    """
    has_dedup = dedup_cfg is not None
    has_reindex = reindex_cfg is not None
    has_cvtlabelme = cvtlabelme_cfg is not None

    if not has_dedup and not has_reindex and not has_cvtlabelme:
        return

    # 仅配置 cvtlabelme 而无 reindex 且无 dedup 时的警告
    if has_cvtlabelme and not has_reindex and not has_dedup:
        print("!" * 90)
        print(
            "警告：cvtlabelme 阶段已配置，但缺少 reindex 阶段。\n"
            "  filter 输出的 YOLO 标签类别 ID 将直接映射到 labelme 类别名称。\n"
            "  如果 ID 未经过重映射，输出 JSON 可能大量出现 'unknown' 标签。\n"
            "  解决方法：添加 [profiles.<名称>.reindex] 节配置重映射。"
        )
        print("!" * 90)

    reindex_script = script_root / "remap_yolo_labels.py"
    cvtlabelme_script = script_root / "yolo_to_labelme.py"
    dedup_script = script_root / "dedup_yolo_dataset.py"

    current_sources = list(filter_output_dirs)

    # ── dedup 阶段 ──────────────────────────────────────────────────────────────────────────
    if has_dedup:
        print("=" * 90)
        print("Pipeline 阶段：dedup（图像去重去相似）")
        print("=" * 90)
        # filter 输出为 merge 根目录，直接作为 scan_dir
        if len(current_sources) != 1:
            print(
                f"[pipeline] 警告：filter 阶段输出了 {len(current_sources)} 个目录，"
                "dedup 阶段期望仅有 1 个 merge 根目录，取第一个目录继续。",
                file=sys.stderr,
            )
        scan_dir = current_sources[0]
        has_subsequent = has_reindex or has_cvtlabelme
        dedup_cmd = build_dedup_command(
            dedup_script,
            scan_dir,
            dedup_cfg,
            print_output_dir=has_subsequent,
        )
        dedup_output_dirs = run_stage(dedup_cmd, "dedup", print_command)
        if has_subsequent:
            if not dedup_output_dirs:
                print(
                    "[pipeline] 警告：dedup 阶段未输出任何 OUTPUT_DIR 行，管道无法继续。",
                    file=sys.stderr,
                )
                sys.exit(1)
            current_sources = dedup_output_dirs

    # ── reindex 阶段 ──────────────────────────────────────────────────────────────────────────
    if has_reindex:
        print("=" * 90)
        print("Pipeline 阶段：reindex（标签序号重映射）")
        print("=" * 90)
        reindex_sources_after: List[Path] = []
        cmds = build_reindex_commands(
            reindex_script,
            current_sources,
            reindex_cfg,
            print_output_dir=has_cvtlabelme,  # 只有 cvtlabelme 后续时才需要输出目录
        )
        for cmd in cmds:
            out_dirs = run_stage(cmd, "reindex", print_command)
            reindex_sources_after.extend(out_dirs)

        # 如果 cvtlabelme 后续且捕获到输出目录，就用新目录；
        # 否则回退到当前源目录（inplace 情况）。
        if reindex_sources_after:
            current_sources = reindex_sources_after

    # ── cvtlabelme 阶段 ────────────────────────────────────────────────────────────────────────
    if has_cvtlabelme:
        print("=" * 90)
        print("Pipeline 阶段：cvtlabelme（转换 labelme JSON）")
        print("=" * 90)
        cmds = build_cvtlabelme_commands(
            cvtlabelme_script,
            current_sources,
            cvtlabelme_cfg,
            print_output_dir=False,
        )
        for cmd in cmds:
            run_stage(cmd, "cvtlabelme", print_command)
