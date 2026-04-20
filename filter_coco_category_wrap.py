#!/usr/bin/env python3
"""
filter_coco_category.py 的配置驱动 wrapper，可选支持流水线阶段。

嵌套 TOML 格式（pipeline 模式）：
    [profiles.<名称>.filter]      # 必选，过滤阶段
    [profiles.<名称>.reindex]     # 可选，过滤后运行 remap_yolo_labels.py
    [profiles.<名称>.cvtlabelme]  # 可选，reindex/filter 后运行 yolo_to_labelme.py

公履层格式（仅过滤）仍向下兼容：
    [profiles.<名称>]
    category_names = "boom"
    ...
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_CONFIG_RELATIVE_PATH = Path("configs/filter_coco_category_profiles.toml")
FILTER_SCRIPT_NAME = "filter_coco_category.py"

FILTER_ALLOWED_KEYS = {
    "category_ids",
    "category_names",
    "json_file",
    "json_dir",
    "label_root",
    "output_root",
    "image_root",
    "dry_run",
    "debug",
    "merge",
}
# 嵌套 pipeline profile 的检测键：任意子表存在即识别为 pipeline 模式
PIPELINE_STAGE_KEYS = {"filter", "reindex", "cvtlabelme", "dedup"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 TOML 加载命名配置并驱动 filter_coco_category.py 执行。"
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


def resolve_input_path(path_str: Optional[str]) -> Path:
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


def get_named_profile(doc: Dict[str, Any], config_name: str) -> Dict[str, Any]:
    profiles = doc.get("profiles")
    if isinstance(profiles, dict) and config_name in profiles:
        profile = profiles[config_name]
        if isinstance(profile, dict):
            return profile

    top_level_profile = doc.get(config_name)
    if isinstance(top_level_profile, dict):
        return top_level_profile

    raise KeyError(
        f"Config profile '{config_name}' not found. Expected [profiles.{config_name}] in TOML."
    )


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


def _normalize_multi_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        items = [x.strip() for x in value.split(",") if x.strip()]
        return ",".join(items) if items else None
    if isinstance(value, list):
        items = [str(x).strip() for x in value if str(x).strip()]
        return ",".join(items) if items else None
    return None


def validate_profile(profile: Dict[str, Any], profile_name: str) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []

    unknown_keys = [k for k in profile.keys() if k not in FILTER_ALLOWED_KEYS]
    if unknown_keys:
        errors.append(
            f"[{profile_name}] unknown keys: {unknown_keys}. Allowed keys: {sorted(FILTER_ALLOWED_KEYS)}"
        )

    category_ids = _normalize_multi_value(profile.get("category_ids"))
    category_names = _normalize_multi_value(profile.get("category_names"))
    has_category_ids = bool(category_ids)
    has_category_names = bool(category_names)
    if has_category_ids == has_category_names:
        errors.append("Exactly one of category_ids or category_names is required")

    if has_category_ids:
        for item in category_ids.split(","):
            if not item.isdigit():
                errors.append(f"category_ids contains non-integer value: {item}")

    json_file = profile.get("json_file")
    json_dir = profile.get("json_dir")
    has_json_file = bool(json_file)
    has_json_dir = bool(json_dir)
    if has_json_file == has_json_dir:
        errors.append("Exactly one of json_file or json_dir is required")

    normalized: Dict[str, Any] = {}

    if has_json_file:
        if not isinstance(json_file, str):
            errors.append("json_file must be string path")
        else:
            path = Path(json_file).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            if not path.is_file():
                errors.append(f"json_file not found: {path}")
            normalized["json_file"] = str(path)

    if has_json_dir:
        if not isinstance(json_dir, str):
            errors.append("json_dir must be string path")
        else:
            path = Path(json_dir).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            if not path.is_dir():
                errors.append(f"json_dir not found: {path}")
            normalized["json_dir"] = str(path)

    for key in ["label_root", "output_root", "image_root"]:
        value = profile.get(key)
        if key in {"label_root", "output_root"} and not value:
            errors.append(f"{key} is required")
            continue
        if value is None:
            continue
        if not isinstance(value, str):
            errors.append(f"{key} must be string path")
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        normalized[key] = str(path)

    merge_value = _to_merge_string(profile.get("merge"))
    if profile.get("merge") is not None and merge_value is None:
        errors.append("merge must be true/false or 'true'/'false'")
    normalized["merge"] = merge_value or "false"

    for key in ["dry_run", "debug"]:
        value = profile.get(key, False)
        if isinstance(value, bool):
            normalized[key] = value
        else:
            errors.append(f"{key} must be boolean")

    if category_ids:
        normalized["category_ids"] = category_ids
    if category_names:
        normalized["category_names"] = category_names

    return normalized, errors


def build_command(validated: Dict[str, Any], print_output_dir: bool = False) -> List[str]:
    cmd: List[str] = [sys.executable, str((script_root() / FILTER_SCRIPT_NAME).resolve())]

    if "category_ids" in validated:
        cmd.extend(["--category-ids", validated["category_ids"]])
    if "category_names" in validated:
        cmd.extend(["--category-names", validated["category_names"]])

    if "json_file" in validated:
        cmd.extend(["--json-file", validated["json_file"]])
    if "json_dir" in validated:
        cmd.extend(["--json-dir", validated["json_dir"]])

    cmd.extend(["--label-root", validated["label_root"]])
    cmd.extend(["--output-root", validated["output_root"]])

    if "image_root" in validated:
        cmd.extend(["--image-root", validated["image_root"]])

    if validated.get("dry_run"):
        cmd.append("--dry-run")
    if validated.get("debug"):
        cmd.append("--debug")

    cmd.extend(["--merge", validated.get("merge", "false")])
    if print_output_dir:
        cmd.append("--print-output-dir")
    return cmd


def is_pipeline_profile(profile: Dict[str, Any]) -> bool:
    """判断该 profile 是否为嵌套 pipeline 格式。

    当 'filter'、'reindex'、'cvtlabelme' 中任意一个为字典子表时返回 True。
    """
    return any(isinstance(profile.get(k), dict) for k in PIPELINE_STAGE_KEYS)


def main() -> int:
    args = parse_args()

    config_path = resolve_input_path(args.config_file)
    try:
        toml_doc = load_toml(config_path)
        raw_profile = get_named_profile(toml_doc, args.config_name)
    except Exception as e:
        print(f"[错误] 加载配置失败：{e}", file=sys.stderr)
        return 1

    # ── 检测 profile 格式 ────────────────────────────────────────────────
    if is_pipeline_profile(raw_profile):
        filter_cfg = raw_profile.get("filter")  # 无 filter 子表时为 None
        reindex_cfg = raw_profile.get("reindex")  # 缺失时为 None
        cvtlabelme_cfg = raw_profile.get("cvtlabelme")  # 缺失时为 None
        dedup_cfg = raw_profile.get("dedup")  # 缺失时为 None
    else:
        # 展平格式（将整个 profile 作为 filter 配置）
        filter_cfg = raw_profile
        reindex_cfg = None
        cvtlabelme_cfg = None
        dedup_cfg = None

    has_filter = filter_cfg is not None
    has_pipeline = reindex_cfg is not None or cvtlabelme_cfg is not None or dedup_cfg is not None

    if has_filter:
        validated, errors = validate_profile(filter_cfg, args.config_name)
        if errors:
            print(f"[ERROR] Config '{args.config_name}' filter section is invalid:", file=sys.stderr)
            for msg in errors:
                print(f"  - {msg}", file=sys.stderr)
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
        print("Pipeline 阶段：filter（filter_coco_category）")
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
        dedup_cfg=dedup_cfg,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
