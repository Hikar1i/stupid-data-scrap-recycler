#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_CONFIG_RELATIVE_PATH = Path("configs/filter_coco_category_profiles.toml")
FILTER_SCRIPT_NAME = "filter_coco_category.py"

ALLOWED_CONFIG_KEYS = {
    "category_id",
    "category_name",
    "json_file",
    "json_dir",
    "label_root",
    "output_root",
    "image_root",
    "dry_run",
    "debug",
    "merge",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load named config from TOML and run filter_coco_category.py with validated args"
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
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print final command before execution",
    )
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


def validate_profile(profile: Dict[str, Any], profile_name: str) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []

    unknown_keys = [k for k in profile.keys() if k not in ALLOWED_CONFIG_KEYS]
    if unknown_keys:
        errors.append(
            f"[{profile_name}] unknown keys: {unknown_keys}. Allowed keys: {sorted(ALLOWED_CONFIG_KEYS)}"
        )

    category_id = profile.get("category_id")
    category_name = profile.get("category_name")
    has_category_id = category_id is not None
    has_category_name = bool(category_name)
    if has_category_id == has_category_name:
        errors.append("Exactly one of category_id or category_name is required")

    if has_category_id and not isinstance(category_id, int):
        errors.append("category_id must be int")
    if has_category_name and not isinstance(category_name, str):
        errors.append("category_name must be string")

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

    if has_category_id and isinstance(category_id, int):
        normalized["category_id"] = category_id
    if has_category_name and isinstance(category_name, str):
        normalized["category_name"] = category_name

    if has_json_file and normalized.get("merge") == "true":
        errors.append("merge=true is only meaningful when using json_dir")

    return normalized, errors


def build_command(validated: Dict[str, Any]) -> List[str]:
    cmd: List[str] = [sys.executable, str((script_root() / FILTER_SCRIPT_NAME).resolve())]

    if "category_id" in validated:
        cmd.extend(["--category-id", str(validated["category_id"])])
    if "category_name" in validated:
        cmd.extend(["--category-name", validated["category_name"]])

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
    return cmd


def main() -> int:
    args = parse_args()

    config_path = resolve_input_path(args.config_file)
    try:
        toml_doc = load_toml(config_path)
        profile = get_named_profile(toml_doc, args.config_name)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}", file=sys.stderr)
        return 1

    validated, errors = validate_profile(profile, args.config_name)
    if errors:
        print(f"[ERROR] Config '{args.config_name}' is invalid:", file=sys.stderr)
        for msg in errors:
            print(f"  - {msg}", file=sys.stderr)
        return 1

    command = build_command(validated)
    if args.print_command:
        print("Command:")
        print(" ".join(command))

    result = subprocess.run(command, cwd=str(script_root()))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
