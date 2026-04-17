#!/usr/bin/env python3
"""
批量扫描 Roboflow 数据集目录，生成 filter_yolo_roboflow_wrap.py 所需的 TOML 配置文件。

用法示例：
    python3 gen_roboflow_filter_config.py \\
        --scan-dir /path/to/datasets \\
        --output-config my_filter.toml \\
        --keywords "crane" \\
        --output-root /path/to/output

    python3 gen_roboflow_filter_config.py \\
        --scan-dir /path/to/datasets \\
        --output-config /abs/path/to/my_filter.toml \\
        --keywords "crane,boom" \\
        --template default_pipeline.toml \\
        --overwrite
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


SCRIPT_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPT_ROOT / "configs"
TEMPLATE_DIR = CONFIGS_DIR / "_template"

# 无模板时的内置默认值
_DEFAULT_FILTER: Dict[str, Any] = {
    "merge": True,
    "dry_run": False,
    "debug": False,
}
_DEFAULT_REINDEX: Dict[str, Any] = {
    "inplace": True,
}
_DEFAULT_CVTLABELME: Dict[str, Any] = {
    "overwrite": False,
}


# ── 参数解析 ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "扫描 Roboflow 数据集目录，按关键词匹配类别，"
            "批量生成 filter_yolo_roboflow_wrap.py 的 TOML 配置文件。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scan-dir",
        required=True,
        metavar="DIR",
        help="扫描该目录下所有包含 data.yaml 的 Roboflow 数据集子目录（仅一层）",
    )
    parser.add_argument(
        "--output-config",
        required=True,
        metavar="FILE",
        help=(
            "输出配置文件路径。仅文件名（不含路径分隔符）时自动放入 configs/ 目录；"
            "含路径分隔符时按绝对/相对路径处理。"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="若目标配置文件已存在则覆盖，默认不覆盖",
    )
    parser.add_argument(
        "--keywords",
        required=True,
        metavar="KW[,KW2,...]",
        help=(
            "类别名匹配关键词，以 *关键词* 方式（子串）匹配。"
            "多个关键词用英文逗号分隔，任一命中即算匹配（OR 逻辑）。"
        ),
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        default=False,
        help="匹配时区分大小写，默认不区分",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        metavar="DIR",
        help=(
            "过滤结果输出目录。"
            "也可在模板文件的 [filter] 节配置 output_root 字段（CLI 参数优先）。"
            "两者必须存在其一，否则报错退出。"
        ),
    )
    parser.add_argument(
        "--template",
        default=None,
        metavar="FILE",
        help=(
            "配置模板文件名（自动在 configs/_template/ 下查找）或绝对/相对路径。"
            "未指定时使用内置默认值并生成完整三阶段（filter+reindex+cvtlabelme）配置。"
            "模板中存在 [reindex] / [cvtlabelme] 节才生成对应阶段。"
        ),
    )
    return parser.parse_args()


# ── 路径解析 ─────────────────────────────────────────────────────────────────

def resolve_output_config_path(name: str) -> Path:
    """
    解析输出配置文件路径。
    不含路径分隔符时放入项目 configs/ 目录；含路径分隔符时按绝对/相对路径处理。
    """
    if "/" not in name and "\\" not in name:
        return CONFIGS_DIR / name
    p = Path(name)
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def resolve_template_path(name: str) -> Path:
    """
    解析模板文件路径。
    仅文件名时在 configs/_template/ 下查找；含路径分隔符时按绝对/相对路径处理。
    """
    p = Path(name)
    if p.is_absolute():
        return p
    if "/" in name or "\\" in name:
        return (Path.cwd() / p).resolve()
    return TEMPLATE_DIR / name


# ── TOML 加载 ─────────────────────────────────────────────────────────────────

def load_toml(path: Path) -> Dict[str, Any]:
    """加载 TOML 文件，兼容 Python 3.11+ 内置 tomllib 和第三方 tomli。"""
    try:
        import tomllib  # type: ignore[import]

        with path.open("rb") as f:
            return tomllib.load(f)
    except ModuleNotFoundError:
        try:
            import tomli  # type: ignore[import]

            with path.open("rb") as f:
                return tomli.load(f)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TOML 解析器不可用，请使用 Python 3.11+ 或安装 tomli（pip install tomli）"
            ) from exc


# ── data.yaml 解析 ─────────────────────────────────────────────────────────────

def load_names_from_yaml(data_yaml: Path) -> Optional[List[str]]:
    """
    从 data.yaml 读取 names 类别列表。
    支持 Roboflow 常见的列表格式和字典格式（{0: 'cls1', 1: 'cls2', ...}）。
    """
    try:
        import yaml  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML 未安装，请执行：pip install pyyaml") from exc

    try:
        with data_yaml.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
    except Exception as e:
        print(f"  [警告] 解析 {data_yaml} 失败：{e}", file=sys.stderr)
        return None

    if not isinstance(doc, dict):
        print(f"  [警告] {data_yaml} 顶层格式非字典", file=sys.stderr)
        return None

    names = doc.get("names")
    if isinstance(names, list):
        return [str(n) for n in names]
    if isinstance(names, dict):
        # 字典格式：{0: 'class1', 1: 'class2', ...}
        return [str(names[k]) for k in sorted(names.keys())]

    print(f"  [警告] {data_yaml} 中未找到有效的 names 字段", file=sys.stderr)
    return None


# ── 目录扫描 ─────────────────────────────────────────────────────────────────

def scan_roboflow_datasets(scan_dir: Path) -> List[Tuple[Path, Path]]:
    """
    扫描目录下一层子目录，返回所有包含 data.yaml 的数据集条目。
    返回值：[(dataset_dir, data_yaml_path), ...]，按目录名字母序排列。
    """
    result: List[Tuple[Path, Path]] = []
    for entry in sorted(scan_dir.iterdir()):
        if not entry.is_dir():
            continue
        candidate = entry / "data.yaml"
        if candidate.is_file():
            result.append((entry, candidate))
    return result


# ── 类别匹配 ─────────────────────────────────────────────────────────────────

def match_categories(
    names: List[str],
    keywords: List[str],
    case_sensitive: bool,
) -> List[Tuple[str, int, int]]:
    """
    从类别名列表中匹配包含关键词的类别，返回三元组列表。

    返回格式：[(类别名, 原始下标, 自增新下标), ...]
      - 原始下标：该类别在 data.yaml names 数组中的原始位置
      - 自增新下标：从 0 开始，按匹配顺序递增（reindex 后的新序号）
      - 一个类别命中多个关键词时只计入一次（OR 逻辑，取 break）
    """
    matched: List[Tuple[str, int, int]] = []
    new_idx = 0
    for orig_idx, name in enumerate(names):
        cmp_name = name if case_sensitive else name.lower()
        for kw in keywords:
            cmp_kw = kw if case_sensitive else kw.lower()
            if cmp_kw in cmp_name:
                matched.append((name, orig_idx, new_idx))
                new_idx += 1
                break  # 同一类别多关键词只计入一次
    return matched


# ── Profile 名称处理 ─────────────────────────────────────────────────────────

def sanitize_profile_name(raw: str) -> str:
    """
    将数据集目录名转换为合法的 TOML 裸键（profile 名称）。
    保留 A-Za-z0-9_-，其余替换为下划线，连续下划线压缩，首尾剔除 _-。
    """
    sanitized = re.sub(r"[^\w-]", "_", raw)     # 替换非法字符为下划线
    sanitized = re.sub(r"_+", "_", sanitized)    # 压缩连续下划线
    sanitized = sanitized.strip("_-")            # 去除首尾
    return sanitized or "dataset"


def make_unique_name(base: str, existing: Set[str]) -> str:
    """若名称已被占用，追加数字后缀确保唯一。"""
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


# ── TOML 字符串生成工具 ─────────────────────────────────────────────────────────

def _toml_bool(v: Any) -> str:
    """将 Python bool 或布尔字符串转换为 TOML 布尔值字面量（true/false）。"""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return "true" if v.strip().lower() == "true" else "false"
    return "false"


def _toml_str(v: str) -> str:
    """生成 TOML 基本字符串字面量，转义反斜杠和双引号。"""
    escaped = v.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


# ── Profile TOML 块构建 ────────────────────────────────────────────────────────

def _build_filter_block(
    profile_name: str,
    data_yaml: Path,
    triples: List[Tuple[str, int, int]],
    output_root: str,
    cfg: Dict[str, Any],
) -> List[str]:
    """生成 [profiles.<name>.filter] TOML 节的行列表。"""
    target_names = ",".join(name for name, _, _ in triples)
    return [
        f"[profiles.{profile_name}.filter]",
        f"data_yaml = {_toml_str(str(data_yaml))}",
        f"target_names = {_toml_str(target_names)}",
        f"output_root = {_toml_str(output_root)}",
        f"merge = {_toml_bool(cfg.get('merge', True))}",
        f"dry_run = {_toml_bool(cfg.get('dry_run', False))}",
        f"debug = {_toml_bool(cfg.get('debug', False))}",
    ]


def _build_reindex_block(
    profile_name: str,
    triples: List[Tuple[str, int, int]],
    cfg: Dict[str, Any],
) -> List[str]:
    """生成 [profiles.<name>.reindex] TOML 节的行列表。"""
    # 格式：原始序号:新序号，如 "3:0,6:1"
    mapping = ",".join(f"{orig}:{new}" for _, orig, new in triples)
    return [
        f"[profiles.{profile_name}.reindex]",
        f"mapping = {_toml_str(mapping)}",
        f"inplace = {_toml_bool(cfg.get('inplace', True))}",
    ]


def _build_cvtlabelme_block(
    profile_name: str,
    triples: List[Tuple[str, int, int]],
    cfg: Dict[str, Any],
) -> List[str]:
    """生成 [profiles.<name>.cvtlabelme] TOML 节的行列表。"""
    # 格式：新序号:类别名，如 "0:crane,1:tower crane"
    mapping = ",".join(f"{new}:{name}" for name, _, new in triples)
    return [
        f"[profiles.{profile_name}.cvtlabelme]",
        f"mapping = {_toml_str(mapping)}",
        f"overwrite = {_toml_bool(cfg.get('overwrite', False))}",
    ]


def build_profile_block(
    profile_name: str,
    data_yaml: Path,
    triples: List[Tuple[str, int, int]],
    output_root: str,
    filter_cfg: Dict[str, Any],
    reindex_cfg: Optional[Dict[str, Any]],
    cvtlabelme_cfg: Optional[Dict[str, Any]],
) -> str:
    """生成单个 profile 的完整 TOML 配置字符串（含阶段间空行）。"""
    lines: List[str] = []

    lines += _build_filter_block(profile_name, data_yaml, triples, output_root, filter_cfg)
    lines.append("")

    if reindex_cfg is not None:
        lines += _build_reindex_block(profile_name, triples, reindex_cfg)
        lines.append("")

    if cvtlabelme_cfg is not None:
        lines += _build_cvtlabelme_block(profile_name, triples, cvtlabelme_cfg)
        lines.append("")

    return "\n".join(lines)


# ── 主逻辑 ────────────────────────────────────────────────────────────────────

def main() -> int:
    """主入口，返回退出码。"""
    args = parse_args()

    # ── 路径解析 ──────────────────────────────────────────────────────────────
    scan_dir = Path(args.scan_dir).expanduser().resolve()
    output_config_path = resolve_output_config_path(args.output_config)
    keywords = [kw.strip() for kw in args.keywords.split(",") if kw.strip()]

    if not scan_dir.is_dir():
        print(f"[错误] 扫描目录不存在或不是目录：{scan_dir}", file=sys.stderr)
        return 1

    if not keywords:
        print("[错误] 关键词不能为空。", file=sys.stderr)
        return 1

    # ── 加载模板 ───────────────────────────────────────────────────────────────
    template_raw: Dict[str, Any] = {}
    template_provided = args.template is not None
    if template_provided:
        template_path = resolve_template_path(args.template)
        if not template_path.is_file():
            print(f"[错误] 模板文件不存在：{template_path}", file=sys.stderr)
            return 1
        try:
            template_raw = load_toml(template_path)
        except Exception as e:
            print(f"[错误] 加载模板失败：{e}", file=sys.stderr)
            return 1
        print(f"[信息] 已加载模板：{template_path}")

    # ── 确定各阶段配置 ─────────────────────────────────────────────────────────
    # filter 阶段：以默认值为基础，由模板覆盖；动态字段（data_yaml/target_names）在此不填
    filter_cfg: Dict[str, Any] = {**_DEFAULT_FILTER, **(template_raw.get("filter") or {})}
    # output_root 从 filter_cfg 中单独提取，不写入通用配置字典
    template_output_root: Optional[str] = filter_cfg.pop("output_root", None)
    if template_output_root is not None and not str(template_output_root).strip():
        template_output_root = None  # 空字符串视为未配置

    # reindex / cvtlabelme 阶段：
    #   有模板时：模板中存在对应节才启用，缺失则该阶段不生成
    #   无模板时：默认启用全三阶段
    if template_provided:
        reindex_cfg: Optional[Dict[str, Any]] = (
            {**_DEFAULT_REINDEX, **(template_raw.get("reindex") or {})}
            if "reindex" in template_raw
            else None
        )
        cvtlabelme_cfg: Optional[Dict[str, Any]] = (
            {**_DEFAULT_CVTLABELME, **(template_raw.get("cvtlabelme") or {})}
            if "cvtlabelme" in template_raw
            else None
        )
    else:
        reindex_cfg = dict(_DEFAULT_REINDEX)
        cvtlabelme_cfg = dict(_DEFAULT_CVTLABELME)

    # ── 确定 output_root（CLI 优先 > 模板 > 报错）─────────────────────────────
    output_root: Optional[str] = args.output_root or template_output_root
    if not output_root:
        print(
            "[错误] 过滤结果输出目录未指定。\n"
            "  请通过 --output-root <目录> 传入，"
            "或在模板文件的 [filter] 节中配置 output_root 字段（两者必须存在其一）。",
            file=sys.stderr,
        )
        return 1

    # ── 检查输出文件冲突 ──────────────────────────────────────────────────────
    if output_config_path.exists() and not args.overwrite:
        print(
            f"[错误] 输出配置文件已存在：{output_config_path}\n"
            "  如需覆盖请添加 --overwrite 参数。",
            file=sys.stderr,
        )
        return 1

    # ── 扫描数据集目录 ─────────────────────────────────────────────────────────
    print(f"[信息] 正在扫描目录：{scan_dir}")
    datasets = scan_roboflow_datasets(scan_dir)
    if not datasets:
        print(
            f"[错误] 在 {scan_dir} 下未找到任何包含 data.yaml 的子目录。",
            file=sys.stderr,
        )
        return 1
    print(f"[信息] 共扫描到 {len(datasets)} 个数据集目录。")
    print(f"[信息] 匹配关键词：{args.keywords}（{'区分' if args.case_sensitive else '不区分'}大小写）")

    # ── 逐个数据集匹配类别并生成 profile ──────────────────────────────────────
    used_names: Set[str] = set()
    profile_blocks: List[str] = []
    skipped = 0

    for dataset_dir, data_yaml in datasets:
        names = load_names_from_yaml(data_yaml)
        if names is None:
            print(f"  [跳过] {dataset_dir.name}（无法读取类别列表）")
            skipped += 1
            continue

        triples = match_categories(names, keywords, args.case_sensitive)
        if not triples:
            print(f"  [跳过] {dataset_dir.name}（无匹配类别）")
            skipped += 1
            continue

        base_name = sanitize_profile_name(dataset_dir.name)
        profile_name = make_unique_name(base_name, used_names)
        used_names.add(profile_name)

        matched_desc = "  |  ".join(
            f"{name}（原始序号 {orig} → 新序号 {new}）"
            for name, orig, new in triples
        )
        print(f"  [命中] {dataset_dir.name}")
        print(f"         profile 名：{profile_name}")
        print(f"         匹配类别：{matched_desc}")

        block = build_profile_block(
            profile_name=profile_name,
            data_yaml=data_yaml,
            triples=triples,
            output_root=output_root,
            filter_cfg=filter_cfg,
            reindex_cfg=reindex_cfg,
            cvtlabelme_cfg=cvtlabelme_cfg,
        )
        profile_blocks.append(block)

    if not profile_blocks:
        print("[错误] 所有数据集均无匹配类别，未生成任何配置。", file=sys.stderr)
        return 1

    # ── 拼接文件内容并写出 ─────────────────────────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    template_display = (
        str(resolve_template_path(args.template)) if args.template else "（内置默认值）"
    )
    header = "\n".join([
        "# 自动生成配置文件",
        f"# 生成时间：{now}",
        f"# 扫描目录：{scan_dir}",
        f"# 匹配关键词：{args.keywords}",
        f"# 配置模板：{template_display}",
        "",
        "[meta]",
        'wrapper_type = "yolo_roboflow"',
        "",
        "",
    ])

    content = header + "\n".join(profile_blocks)

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(content, encoding="utf-8")

    skip_note = f"，跳过 {skipped} 个数据集" if skipped else ""
    print(
        f"\n[完成] 已生成 {len(profile_blocks)} 个配置段{skip_note}，"
        f"写入：{output_config_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
