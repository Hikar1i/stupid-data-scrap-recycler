#!/usr/bin/env python3
"""
批量执行 wrapper 脚本（filter_yolo_roboflow_wrap.py 或 filter_coco_category_wrap.py）
对 TOML 配置文件中所有有效配置段逐一运行。

使用方式：
  python3 batch_run_profiles.py --wrapper filter_yolo_roboflow_wrap.py \\
      --config-file configs/filter_yolo_roboflow_profiles.toml

可选参数：
  --dry-run        仅列出配置段，不实际执行
  --print-command  透传给各 wrapper 子进程，执行前打印最终命令
  --fail-fast      遇到首个失败配置段时立即终止（默认：继续执行全部）
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# wrapper 文件名与 [meta].wrapper_type 值的对应关系
WRAPPER_TYPE_TO_SCRIPT: Dict[str, str] = {
    "yolo_roboflow": "filter_yolo_roboflow_wrap.py",
    "coco_category": "filter_coco_category_wrap.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量执行 wrapper 脚本的所有配置段。"
    )
    parser.add_argument(
        "--wrapper",
        required=True,
        metavar="SCRIPT",
        help="wrapper 脚本的文件名或路径，例如 filter_yolo_roboflow_wrap.py",
    )
    parser.add_argument(
        "--config-file",
        required=True,
        metavar="TOML",
        help="TOML 配置文件路径（绝对或相对）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式：列出所有配置段但不执行任何命令",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="透传 --print-command 给每个 wrapper 子进程，执行前打印最终命令",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="遇到首个失败的配置段时立即终止（默认：继续执行所有配置段）",
    )
    return parser.parse_args()


def script_root() -> Path:
    """返回本脚本所在目录。"""
    return Path(__file__).resolve().parent


def load_toml(path: Path) -> Dict[str, Any]:
    """加载并解析 TOML 文件，优先使用标准库 tomllib（Python 3.11+），回退到 tomli。"""
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    try:
        import tomllib  # type: ignore[attr-defined]
        with path.open("rb") as f:
            return tomllib.load(f)
    except ModuleNotFoundError:
        pass
    try:
        import tomli  # type: ignore
        with path.open("rb") as f:
            return tomli.load(f)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TOML 解析器不可用，请使用 Python 3.11+ 或安装 tomli（pip install tomli）。"
        ) from exc


def resolve_wrapper_path(wrapper_arg: str) -> Path:
    """将 --wrapper 参数解析为脚本绝对路径。

    查找顺序：
      1. 绝对路径 → 直接使用
      2. 相对路径 → 先尝试脚本同目录，再尝试当前工作目录
    """
    raw = Path(wrapper_arg)
    if raw.is_absolute():
        return raw.resolve()
    candidate = (script_root() / raw).resolve()
    if candidate.is_file():
        return candidate
    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.is_file():
        return cwd_candidate
    # 返回同目录候选路径，后续统一报错
    return candidate


def enumerate_profiles(doc: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """枚举 TOML 文档中所有配置段名称，区分有效与无效。

    有效：[profiles.<name>] 的值为 dict。
    无效：[profiles.<name>] 的值不为 dict（作为防御性检查，TOML 中罕见）。
    返回 (有效名称列表, 无效名称列表)。
    """
    profiles_section = doc.get("profiles", {})
    if not isinstance(profiles_section, dict):
        return [], []
    valid: List[str] = []
    invalid: List[str] = []
    for name, value in profiles_section.items():
        if isinstance(value, dict):
            valid.append(name)
        else:
            invalid.append(name)
    return valid, invalid


def check_wrapper_type(doc: Dict[str, Any], wrapper_path: Path) -> Optional[str]:
    """校验 [meta].wrapper_type 与实际 wrapper 脚本是否匹配。

    返回值：
      None          → 匹配通过
      非空字符串     → 警告或错误描述（含"[错误]"前缀表示硬性不匹配）
    """
    meta = doc.get("meta")
    if not isinstance(meta, dict):
        return "[警告] 配置文件中无 [meta] 节，跳过类型校验。"
    wrapper_type = meta.get("wrapper_type")
    if not wrapper_type:
        return "[警告] [meta] 节中无 wrapper_type 字段，跳过类型校验。"
    expected_script = WRAPPER_TYPE_TO_SCRIPT.get(str(wrapper_type))
    if expected_script is None:
        known = list(WRAPPER_TYPE_TO_SCRIPT.keys())
        return f"[警告] [meta].wrapper_type 值未知：'{wrapper_type}'（已知类型：{known}），跳过类型校验。"
    if wrapper_path.name != expected_script:
        return (
            f"[错误] wrapper 类型不匹配！\n"
            f"  配置文件声明类型：{wrapper_type}（期望脚本：{expected_script}）\n"
            f"  实际指定脚本：{wrapper_path.name}"
        )
    return None


def run_profile(
    wrapper_path: Path,
    config_file: Path,
    profile_name: str,
    print_command: bool,
) -> int:
    """执行单个配置段，实时流式输出子进程内容，返回退出码。"""
    cmd = [
        sys.executable,
        str(wrapper_path),
        "--config-file", str(config_file),
        "--config-name", profile_name,
    ]
    if print_command:
        cmd.append("--print-command")
    result = subprocess.run(cmd)
    return result.returncode


def main() -> int:
    args = parse_args()

    config_file = Path(args.config_file).expanduser().resolve()
    wrapper_path = resolve_wrapper_path(args.wrapper)

    # ── 前置检查 ──────────────────────────────────────────────────────────────
    if not wrapper_path.is_file():
        print(f"[错误] wrapper 脚本不存在：{wrapper_path}", file=sys.stderr)
        return 1

    print("=" * 70)
    print(f"  wrapper    : {wrapper_path.name}")
    print(f"  配置文件   : {config_file}")
    print("=" * 70)

    # ── 加载 TOML ─────────────────────────────────────────────────────────────
    try:
        doc = load_toml(config_file)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[错误] TOML 加载失败：{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[错误] TOML 解析失败（语法错误）：{exc}", file=sys.stderr)
        return 1

    # ── 类型校验 ──────────────────────────────────────────────────────────────
    type_msg = check_wrapper_type(doc, wrapper_path)
    if type_msg is not None:
        print(type_msg, file=sys.stderr if type_msg.startswith("[错误]") else sys.stdout)
        if type_msg.startswith("[错误]"):
            return 1
    else:
        wrapper_type = doc["meta"]["wrapper_type"]
        print(f"[OK] 类型校验通过：wrapper_type = {wrapper_type}")

    # ── 枚举配置段 ────────────────────────────────────────────────────────────
    valid_profiles, invalid_profiles = enumerate_profiles(doc)

    print(f"\n配置段总览：有效 {len(valid_profiles)} 个，无效 {len(invalid_profiles)} 个")
    for name in valid_profiles:
        print(f"  [有效] {name}")
    for name in invalid_profiles:
        print(f"  [无效] {name}（值不为字典，已跳过）")

    if not valid_profiles:
        print("\n[提示] 无有效配置段可执行。")
        return 0

    if args.dry_run:
        print("\n[dry-run] 预览模式，不执行任何命令。")
        return 0

    # ── 批量执行 ──────────────────────────────────────────────────────────────
    total = len(valid_profiles)
    print(f"\n{'=' * 70}")
    print(f"开始批量执行，共 {total} 个配置段")
    print("=" * 70)

    success_names: List[str] = []
    failed_names: List[str] = []

    for idx, name in enumerate(valid_profiles, 1):
        print(f"\n[{idx}/{total}] 配置段：{name}")
        print("-" * 70)

        rc = run_profile(wrapper_path, config_file, name, args.print_command)

        if rc == 0:
            success_names.append(name)
            print(f"[{idx}/{total}] 完成：{name}")
        else:
            failed_names.append(name)
            print(f"[{idx}/{total}] 失败：{name}（退出码 {rc}）", file=sys.stderr)
            if args.fail_fast:
                print("[fail-fast] 检测到失败，终止后续执行。", file=sys.stderr)
                break

    # ── 汇总报告 ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("批量执行完成")
    print(f"  成功：{len(success_names)} 个")
    for name in success_names:
        print(f"    {name}")
    if failed_names:
        print(f"  失败：{len(failed_names)} 个")
        for name in failed_names:
            print(f"    {name}")
    skipped = total - len(success_names) - len(failed_names)
    if skipped > 0:
        print(f"  未执行（fail-fast 截断）：{skipped} 个")
    print("=" * 70)

    return 1 if failed_names else 0


if __name__ == "__main__":
    raise SystemExit(main())
