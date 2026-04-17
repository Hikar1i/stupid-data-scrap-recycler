#!/usr/bin/env python3
"""
filter_yolo_roboflow.py 的配置驱动 wrapper，可选支持流水线阶段。

嵌套 TOML 格式（pipeline 模式）：
    [profiles.<名称>.filter]      # 必选，过滤阶段
    [profiles.<名称>.reindex]     # 可选，过滤后运行 remap_yolo_labels.py
    [profiles.<名称>.cvtlabelme]  # 可选，reindex/filter 后运行 yolo_to_labelme.py

展平格式（仅过滤）仍向下兼容：
    [profiles.<名称>]
    target_names = "Excavator"
    ...
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_CONFIG_RELATIVE_PATH = Path("configs/filter_yolo_roboflow_profiles.toml")
TARGET_SCRIPT_NAME = "filter_yolo_roboflow.py"

FILTER_ALLOWED_KEYS = {
    "data_yaml",
    "target_names",
    "target_ids",
    "output_root",
    "merge",
    "train_images",
    "val_images",
    "test_images",
    "train_labels",
    "val_labels",
    "test_labels",
    "dry_run",
    "debug",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 TOML 加载命名配置并驱动 filter_yolo_roboflow.py 执行。"
    )
    parser.add_argument("--config-name", required=True, help="配置文件中的 profile 名称")
    parser.add_argument(
        "--config-file",
        default=None,
        help=(
            "TOML 配置文件路径（绝对或相对）。省略时使用默认路径："
            f"{DEFAULT_CONFIG_RELATIVE_PATH}"
        ),
    )
    parser.add_argument("--print-command", action="store_true", help="执行前打印最终命令")
    return parser.parse_args()


def script_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_config_path(path_str: Optional[str]) -> Path:
    if not path_str:
        return (script_root() / DEFAULT_CONFIG_RELATIVE_PATH).resolve()

    raw = Path(path_str).expanduser()
    if raw.is_absolute():
        return raw.resolve()

    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (script_root() / raw).resolve()


def load_toml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        import tomllib  # type: ignore[attr-defined]

        with path.open("rb") as f:
            return tomllib.load(f)
    except ModuleNotFoundError:
        try:
            import tomli  # type: ignore

            with path.open("rb") as f:
                return tomli.load(f)
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "TOML parser not available. Use Python 3.11+ or install tomli (pip install tomli)."
            ) from e


def get_profile(doc: Dict[str, Any], name: str) -> Dict[str, Any]:
    profiles = doc.get("profiles")
    if isinstance(profiles, dict) and name in profiles and isinstance(profiles[name], dict):
        return profiles[name]

    top_level = doc.get(name)
    if isinstance(top_level, dict):
        return top_level

    raise KeyError(f"Config profile '{name}' not found. Expected [profiles.{name}]")


def _normalize_multi_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [x.strip() for x in value.split(",") if x.strip()]
        return ",".join(parts) if parts else None
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return ",".join(parts) if parts else None
    return None


def _to_merge_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered
    return None


def _normalize_path(value: Any, key: str, required: bool, errors: List[str]) -> Optional[str]:
    if value is None:
        if required:
            errors.append(f"{key} is required")
        return None
    if not isinstance(value, str):
        errors.append(f"{key} must be string path")
        return None

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return str(path)


def validate_profile(profile: Dict[str, Any], profile_name: str) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []

    unknown = [k for k in profile.keys() if k not in FILTER_ALLOWED_KEYS]
    if unknown:
        errors.append(f"[{profile_name}] unknown keys: {unknown}")

    normalized: Dict[str, Any] = {}

    normalized["data_yaml"] = _normalize_path(profile.get("data_yaml"), "data_yaml", True, errors)
    normalized["output_root"] = _normalize_path(profile.get("output_root"), "output_root", True, errors)

    target_names = _normalize_multi_value(profile.get("target_names"))
    target_ids = _normalize_multi_value(profile.get("target_ids"))
    if bool(target_names) == bool(target_ids):
        errors.append("Exactly one of target_names or target_ids is required")
    if target_ids:
        for item in target_ids.split(","):
            if not item.isdigit():
                errors.append(f"target_ids contains non-integer value: {item}")
    normalized["target_names"] = target_names
    normalized["target_ids"] = target_ids

    merge = _to_merge_string(profile.get("merge"))
    if profile.get("merge") is not None and merge is None:
        errors.append("merge must be true/false or 'true'/'false'")
    normalized["merge"] = merge or "false"

    for key in [
        "train_images",
        "val_images",
        "test_images",
        "train_labels",
        "val_labels",
        "test_labels",
    ]:
        val = profile.get(key)
        if val is None:
            continue
        normalized[key] = _normalize_path(val, key, False, errors)

    for key in ["dry_run", "debug"]:
        val = profile.get(key, False)
        if isinstance(val, bool):
            normalized[key] = val
        else:
            errors.append(f"{key} must be boolean")

    if normalized.get("data_yaml") and not Path(normalized["data_yaml"]).is_file():
        errors.append(f"data_yaml not found: {normalized['data_yaml']}")

    for key in [
        "train_images",
        "val_images",
        "test_images",
        "train_labels",
        "val_labels",
        "test_labels",
    ]:
        value = normalized.get(key)
        if value and not Path(value).is_dir():
            errors.append(f"{key} not found or not directory: {value}")

    return normalized, errors


def build_command(validated: Dict[str, Any], print_output_dir: bool = False) -> List[str]:
    cmd: List[str] = [sys.executable, str((script_root() / TARGET_SCRIPT_NAME).resolve())]

    cmd.extend(["--data-yaml", validated["data_yaml"]])
    cmd.extend(["--output-root", validated["output_root"]])

    if validated.get("target_names"):
        cmd.extend(["--target-names", validated["target_names"]])
    if validated.get("target_ids"):
        cmd.extend(["--target-ids", validated["target_ids"]])

    cmd.extend(["--merge", validated.get("merge", "false")])

    key_to_arg = {
        "train_images": "--train-images",
        "val_images": "--val-images",
        "test_images": "--test-images",
        "train_labels": "--train-labels",
        "val_labels": "--val-labels",
        "test_labels": "--test-labels",
    }

    for key, arg in key_to_arg.items():
        if validated.get(key):
            cmd.extend([arg, validated[key]])

    if validated.get("dry_run"):
        cmd.append("--dry-run")
    if validated.get("debug"):
        cmd.append("--debug")
    if print_output_dir:
        cmd.append("--print-output-dir")

    return cmd


PIPELINE_STAGE_KEYS = {"filter", "reindex", "cvtlabelme"}


def is_pipeline_profile(profile: Dict[str, Any]) -> bool:
    """判断该 profile 是否为嵌套 pipeline 格式。

    当 'filter'、'reindex'、'cvtlabelme' 中任意一个为字典子表时返回 True。
    """
    return any(isinstance(profile.get(k), dict) for k in PIPELINE_STAGE_KEYS)


def main() -> int:
    args = parse_args()

    config_path = resolve_config_path(args.config_file)
    try:
        doc = load_toml(config_path)
        raw_profile = get_profile(doc, args.config_name)
    except Exception as e:
        print(f"[错误] 加载配置失败：{e}", file=sys.stderr)
        return 1

    # ── 检测 profile 格式 ────────────────────────────────────────────────
    if is_pipeline_profile(raw_profile):
        filter_cfg = raw_profile.get("filter")  # 无 filter 子表时为 None
        reindex_cfg = raw_profile.get("reindex")  # 缺失时为 None
        cvtlabelme_cfg = raw_profile.get("cvtlabelme")  # 缺失时为 None
    else:
        # 展平格式（将整个 profile 作为 filter 配置）
        filter_cfg = raw_profile
        reindex_cfg = None
        cvtlabelme_cfg = None

    has_filter = filter_cfg is not None
    has_pipeline = reindex_cfg is not None or cvtlabelme_cfg is not None

    if has_filter:
        validated, errors = validate_profile(filter_cfg, args.config_name)
        if errors:
            print(f"[ERROR] Config '{args.config_name}' filter section is invalid:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return 1
        command = build_command(validated, print_output_dir=has_pipeline)
    else:
        command = None

    if not has_filter and not has_pipeline:
        print("[错误] Profile 中无有效的阶段节（filter/reindex/cvtlabelme）。", file=sys.stderr)
        return 1

    if args.print_command and command:
        print("最终命令：")
        print(" ".join(command))

    if not has_pipeline:
        # 仅过滤模式直接执行
        result = subprocess.run(command, cwd=str(script_root()))
        return result.returncode

    # ── 执行管道 ─────────────────────────────────────────────────────────
    from pipeline_utils import run_stage, run_pipeline_stages  # type: ignore

    filter_output_dirs: List[Path] = []
    if has_filter:
        print("=" * 90)
        print("Pipeline 阶段：filter（filter_yolo_roboflow）")
        print("=" * 90)
        filter_output_dirs = run_stage(command, "filter", args.print_command)
        if not filter_output_dirs:
            print(
                "[pipeline] 警告：filter 阶段未输出任何 OUTPUT_DIR 行，"
                "管道无法继续。",
                file=sys.stderr,
            )
            return 1
    else:
        # 无 filter 阶段，第一个阶段配置中必须有 source_dir
        first_cfg = reindex_cfg if reindex_cfg is not None else cvtlabelme_cfg
        source_dir_raw = first_cfg.get("source_dir") if first_cfg else None
        if not source_dir_raw:
            print(
                "[错误] 无 [filter] 节且第一个管道阶段配置中无 'source_dir' 字段。",
                file=sys.stderr,
            )
            return 1
        filter_output_dirs = [Path(source_dir_raw).expanduser().resolve()]

    run_pipeline_stages(
        filter_output_dirs=filter_output_dirs,
        script_root=script_root(),
        reindex_cfg=reindex_cfg,
        cvtlabelme_cfg=cvtlabelme_cfg,
        print_command=args.print_command,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
