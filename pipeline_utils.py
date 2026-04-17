"""流水线 wrapper 脚本共享工具模块。

管道阶段：filter → reindex → cvtlabelme
每个阶段从上一阶段的 stdout 中解析 OUTPUT_DIR:<路径> 行，
作为下一阶段的源目录传入。
"""
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


OUTPUT_DIR_PREFIX = "OUTPUT_DIR:"


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
) -> None:
    """在 filter 阶段完成后，依次调度 reindex 和或 cvtlabelme 阶段。

    filter_output_dirs：filter 阶段产生的输出目录列表。
    """
    has_reindex = reindex_cfg is not None
    has_cvtlabelme = cvtlabelme_cfg is not None

    if not has_reindex and not has_cvtlabelme:
        return

    # 仅配置 cvtlabelme 而无 reindex 时的警告
    if has_cvtlabelme and not has_reindex:
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

    current_sources = list(filter_output_dirs)

    # ── reindex 阶段 ─────────────────────────────────────────────────────────
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
        # 否则回退到 filter 输出目录（inplace 情况）。
        if reindex_sources_after:
            current_sources = reindex_sources_after
        # inplace=true 时 reindex 已在 filter 输出目录内原地修改，current_sources 保持不变。

    # ── cvtlabelme 阶段 ───────────────────────────────────────────────────────
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
