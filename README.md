# COCO Category Filter & Copy Script

该项目提供一个脚本 `filter_coco_category.py`，用于从 COCO 标注中按类别筛选图像，并将图像与对应标签拷贝到目标目录。

## 功能用途

- 支持按 `category_id` 或 `category_name` 筛选目标类别（两者二选一）。
- 支持处理单个 JSON（`--json-file`）或目录下多个 JSON（`--json-dir`，会读取该目录下所有 `*.json`）。
- 从 COCO 的 `annotations` 中筛选匹配类别的 `image_id`，去重后在 `images` 中提取 `file_name`。
- 根据 `file_name` 生成标签文件路径：
  - 规则：`<label_root>/<file_name父目录名>/<file_name去后缀>.txt`
- 拷贝结果输出到 `output_root` 下新建目录，并自动创建 `images/` 与 `labels/` 子目录。
- 支持 `--dry-run` 预览（不创建目录、不拷贝）。
- 支持 `--debug` 打印每个匹配图像路径及其标签路径。
- 支持 `--merge` 控制 `json-dir` 场景下多 JSON 合并或分开输出。

## 运行环境

- Python 3.8+
- 仅依赖 Python 标准库，无需额外安装第三方包

> 说明：`filter_coco_category_wrap.py` 读取 TOML 配置时优先使用 Python 3.11+ 的 `tomllib`。若低版本 Python 运行 wrapper，请先安装 `tomli`：`pip install tomli`。

## 参数说明

### 必选参数

- `--category-id <int>` / `--category-name <str>`
  - 二选一，指定目标类别。
- `--json-file <path>` / `--json-dir <path>`
  - 二选一，指定单个 COCO JSON 或 JSON 目录。
- `--label-root <path>`
  - 标签根目录。
- `--output-root <path>`
  - 拷贝输出根目录。

### 可选参数

- `--image-root <path>`
  - 可选图像根目录。
  - 当 `file_name` 为相对路径时：
    - 若提供 `--image-root`：按 `<image_root>/<file_name>` 解析图像路径；
    - 若未提供：按 `<json文件所在目录>/<file_name>` 解析。
- `--dry-run`
  - 预览模式：仅输出统计和目标路径，不执行目录创建和文件拷贝。
- `--debug`
  - 调试模式：输出每个匹配项的 `file_name`（解析后的图像路径）和对应标签绝对路径。
- `--merge true|false`（默认 `false`）
  - 仅在 `--json-dir` 模式有意义：
    - `true`：多个 JSON 的结果合并到一个目录；
    - `false`：每个 JSON 单独输出到各自目录。

## 输出目录命名规则

时间戳格式：`%y%m%d_%H%M%S`，如 `260413_183601`。

- `--json-file`：
  - 目录名：`[json文件前缀]_[category_name]_[时间戳]`
  - 示例：`train_person_260413_183601`
- `--json-dir --merge false`：
  - 每个 JSON 各自生成目录：
  - 例如 `train_person_...`、`val_person_...`
- `--json-dir --merge true`：
  - 合并为一个目录：`merge_[category_name]_[时间戳]`

每个输出目录结构如下：

```text
<run_dir>/
  images/
  labels/
```

## 使用示例

### 1) 单文件 + 按类别名筛选

```bash
python3 filter_coco_category.py \
  --category-name person \
  --json-file /path/to/train.json \
  --label-root /path/to/labels/yolo/object_model \
  --output-root /path/to/output
```

### 2) 单文件 + 按类别 ID 筛选

```bash
python3 filter_coco_category.py \
  --category-id 1 \
  --json-file /path/to/val.json \
  --label-root /path/to/labels/yolo/object_model \
  --output-root /path/to/output
```

### 3) 目录模式 + 不合并（每个 JSON 单独目录）

```bash
python3 filter_coco_category.py \
  --category-name person \
  --json-dir /path/to/coco/jsons \
  --label-root /path/to/labels/yolo/object_model \
  --output-root /path/to/output \
  --merge false
```

### 4) 目录模式 + 合并输出

```bash
python3 filter_coco_category.py \
  --category-name person \
  --json-dir /path/to/coco/jsons \
  --label-root /path/to/labels/yolo/object_model \
  --output-root /path/to/output \
  --merge true
```

### 5) 预览路径与统计（不拷贝）

```bash
python3 filter_coco_category.py \
  --category-name person \
  --json-dir /path/to/coco/jsons \
  --label-root /path/to/labels/yolo/object_model \
  --output-root /path/to/output \
  --merge true \
  --dry-run
```

### 6) 调试模式（逐条打印映射）

```bash
python3 filter_coco_category.py \
  --category-name person \
  --json-file /path/to/train.json \
  --label-root /path/to/labels/yolo/object_model \
  --output-root /path/to/output \
  --debug
```

## Wrapper（配置驱动运行）

项目提供了包装脚本：`filter_coco_category_wrap.py`。

目的：通过“配置文件 + 配置名”加载参数，再调用 `filter_coco_category.py`，避免把原脚本逻辑改复杂。

### 配置格式选择

- 当前选择：`TOML`
- 原因：结构化清晰、可维护性高、支持多组命名配置、适合该场景

### 默认配置文件

- 默认路径：`configs/filter_coco_category_profiles.toml`
- 该命名是脚本专用，不使用泛化的 `default.toml`，便于未来扩展其他脚本配置。

### 加载规则

- 仅指定 `--config-name`：从默认配置文件中读取该配置。
- 同时指定 `--config-file` 与 `--config-name`：从指定配置文件读取。
- `--config-file` 同时支持：
  - 绝对路径
  - 相对路径（优先按当前工作目录解析，找不到则按项目根目录解析）

### 配置校验规则

wrapper 会先校验配置，校验不通过则**不会调用** `filter_coco_category.py`：

- `category_id` / `category_name` 必须二选一
- `json_file` / `json_dir` 必须二选一
- `label_root`、`output_root` 必填
- `merge` 必须是 `true/false`（布尔或字符串均可）
- `dry_run`、`debug` 必须为布尔
- 路径项会检查存在性（`json_file`、`json_dir`）

### Wrapper 参数

- `--config-name <name>`：必填，配置名
- `--config-file <path>`：可选，配置文件路径
- `--print-command`：可选，打印最终调用命令

### 配置文件示例

默认文件 `configs/filter_coco_category_profiles.toml` 中可写多组配置：

```toml
[profiles.demo_by_name]
category_name = "person"
json_dir = "/path/to/coco/jsons"
label_root = "/path/to/labels/yolo/object_model"
output_root = "/path/to/output"
merge = "false"
dry_run = true
debug = false

[profiles.demo_by_id]
category_id = 1
json_file = "/path/to/train.json"
label_root = "/path/to/labels/yolo/object_model"
output_root = "/path/to/output"
merge = "false"
dry_run = false
debug = false
```

### Wrapper 使用示例

1) 使用默认配置文件：

```bash
python3 filter_coco_category_wrap.py --config-name demo_by_name
```

2) 使用指定配置文件（相对路径）：

```bash
python3 filter_coco_category_wrap.py \
  --config-file ./configs/filter_coco_category_profiles.toml \
  --config-name demo_by_id
```

3) 使用指定配置文件（绝对路径）并打印最终命令：

```bash
python3 filter_coco_category_wrap.py \
  --config-file /abs/path/to/filter_profiles.toml \
  --config-name demo_by_name \
  --print-command
```

## 关键逻辑说明

1. 在 `categories` 中找到目标类别（按 `id` 或 `name`）。
2. 在 `annotations` 中筛选匹配类别的记录并收集 `image_id`（去重）。
3. 在 `images` 中根据 `image_id` 收集 `file_name`。
4. 根据 `file_name` 构建：
   - 图像源路径
   - 标签源路径：`<label_root>/<parent>/<stem>.txt`
5. 按 `--merge` 策略组织输出任务并执行（或 dry-run 预览）。
6. 输出每个 JSON 的匹配统计和每个输出任务的拷贝统计。
