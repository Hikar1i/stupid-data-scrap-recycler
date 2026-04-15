#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_CONFIG_RELATIVE_PATH = Path("configs/filter_yolo_roboflow_profiles.toml")
TARGET_SCRIPT_NAME = "filter_yolo_roboflow.py"

ALLOWED_KEYS = {
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
        description=(
            "Load named config from TOML and run filter_yolo_roboflow.py with validated args"
        )
    )
    parser.add_argument("--config-name", required=True, help="Profile name in config file")
    parser.add_argument(
        "--config-file",
        default=None,
        help=(
            "Config TOML path (absolute or relative). If omitted, use default: "
            f"{DEFAULT_CONFIG_RELATIVE_PATH}"
        ),
    )
    parser.add_argument("--print-command", action="store_true", help="Print command before execution")
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

    unknown = [k for k in profile.keys() if k not in ALLOWED_KEYS]
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


def build_command(validated: Dict[str, Any]) -> List[str]:
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

    return cmd


def main() -> int:
    args = parse_args()

    config_path = resolve_config_path(args.config_file)
    try:
        doc = load_toml(config_path)
        profile = get_profile(doc, args.config_name)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}", file=sys.stderr)
        return 1

    validated, errors = validate_profile(profile, args.config_name)
    if errors:
        print(f"[ERROR] Config '{args.config_name}' is invalid:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    command = build_command(validated)
    if args.print_command:
        print("Command:")
        print(" ".join(command))

    result = subprocess.run(command, cwd=str(script_root()))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
