# COCO Category Filter & Copy Script

该项目提供一个脚本 `filter_coco_category.py`，用于从 COCO 标注中按类别筛选图像，并将图像与对应标签拷贝到目标目录。

## 功能用途

- 支持按 `category_ids` 或 `category_names` 输入目标类别（两者二选一，支持逗号分隔多类别）。
- 支持处理单个 JSON（`--json-file`）或目录下多个 JSON（`--json-dir`，会读取该目录下所有 `*.json`）。
- 从 COCO 的 `annotations` 中筛选匹配类别的 `image_id`，去重后在 `images` 中提取 `file_name`。
- 根据 `file_name` 生成标签文件路径：
  - 规则：`<label_root>/<file_name父目录名>/<file_name去后缀>.txt`
- 拷贝时会过滤标签行，输出标签文件仅保留目标类别 id 的行（不修改原始标签文件）。
- **COCO→YOLO ID 映射**：COCO 的 `category_id` 从 1 开始，YOLO 标签首列 id 从 0 开始，脚本自动按 `yolo_label_id = coco_category_id - 1` 映射后再过滤标签行，并在输出中打印映射提示和两套 id 对照。
- 拷贝结果输出到 `output_root` 下新建目录，并自动创建 `<json前缀>/images` 与 `<json前缀>/labels` 子目录。
- 支持 `--dry-run` 预览（不创建目录、不拷贝）。
- 支持 `--debug` 打印每个匹配图像路径及其标签路径。
- 支持 `--merge` 控制“多类别”输出：合并到一个目录或每个类别单独目录。

## 运行环境

- Python 3.8+
- 仅依赖 Python 标准库，无需额外安装第三方包

> 说明：`filter_coco_category_wrap.py` 读取 TOML 配置时优先使用 Python 3.11+ 的 `tomllib`。若低版本 Python 运行 wrapper，请先安装 `tomli`：`pip install tomli`。

## 参数说明

### 必选参数

- `--category-ids <id1,id2,...>` / `--category-names <name1,name2,...>`
  - 二选一，支持多类别（逗号分隔）。
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
  - 多类别输出模式：
    - `true`：多个类别结果合并到一个目录；
    - `false`：每个类别单独目录。

## 输出目录命名规则

时间戳格式：`%y%m%d_%H%M%S`，如 `260413_183601`。

- 多类别且 `--merge false`：
  - 每个类别目录：`[category_name]_[时间戳]`
- 多类别且 `--merge true`：
  - 合并目录：`merge_[时间戳]`

无论 `--json-file` 还是 `--json-dir`，都在 run 目录下按每个 JSON 的前缀继续分层：

- `<run_dir>/<json_prefix>/images`
- `<run_dir>/<json_prefix>/labels`

每个输出目录结构如下：

```text
<run_dir>/
  <json_prefix>/
    images/
    labels/
```

## 使用示例

### 1) 单文件 + 按类别名筛选

```bash
python3 filter_coco_category.py \
  --category-names person \
  --json-file /path/to/train.json \
  --label-root /path/to/labels/yolo/object_model \
  --output-root /path/to/output
```

### 2) 单文件 + 按类别 ID 筛选

```bash
python3 filter_coco_category.py \
  --category-ids 1 \
  --json-file /path/to/val.json \
  --label-root /path/to/labels/yolo/object_model \
  --output-root /path/to/output
```

### 3) 多类别 + 不合并（每个类别单独目录）

```bash
python3 filter_coco_category.py \
  --category-names boom,fence \
  --json-dir /path/to/coco/jsons \
  --label-root /path/to/labels/yolo/object_model \
  --output-root /path/to/output \
  --merge false
```

### 4) 多类别 + 合并输出

```bash
python3 filter_coco_category.py \
  --category-names boom,fence \
  --json-dir /path/to/coco/jsons \
  --label-root /path/to/labels/yolo/object_model \
  --output-root /path/to/output \
  --merge true
```

### 5) 预览路径与统计（不拷贝）

```bash
python3 filter_coco_category.py \
  --category-names person \
  --json-dir /path/to/coco/jsons \
  --label-root /path/to/labels/yolo/object_model \
  --output-root /path/to/output \
  --merge true \
  --dry-run
```

### 6) 调试模式（逐条打印映射）

```bash
python3 filter_coco_category.py \
  --category-names person \
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

- `category_ids` / `category_names` 必须二选一（支持逗号分隔多值）
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
category_names = "person"
json_dir = "/path/to/coco/jsons"
label_root = "/path/to/labels/yolo/object_model"
output_root = "/path/to/output"
merge = "false"
dry_run = true
debug = false

[profiles.demo_by_id]
category_ids = "1"
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

## Roboflow YOLO 过滤脚本

新增脚本：`filter_yolo_roboflow.py`

用途：过滤 Roboflow 下载的 YOLO 数据集（`data.yaml`），按指定类别提取图像和标签，并输出为新的 YOLO 目录结构。

### 输入参数

- `--data-yaml <path>`（必填）
  - Roboflow 的 `data.yaml` 路径。
- `--target-names <n1,n2,...>` / `--target-ids <i1,i2,...>`（二选一，必填）
  - 支持多个类别，逗号分隔。
- `--output-root <path>`（必填）
  - 输出根目录。
- `--merge true|false`（可选，默认 `false`）
  - 多类别时是否合并输出。
- 可选覆盖路径：
  - `--train-images` `--val-images` `--test-images`
  - `--train-labels` `--val-labels` `--test-labels`
- `--dry-run`
  - 仅预览，不创建目录、不拷贝、不写标签。
- `--debug`
  - 输出详细路径与逐文件处理信息。

### 路径解析规则

- 若未指定 `--train/val/test-images`，默认按 `data.yaml` 中 `train/val/test` 解析。
- Roboflow 常见相对路径（如 `../train/images`）会优先尝试解析为与 `data.yaml` 同级的 `train/images`。
- 若未指定 `--*-labels`，默认从对应图片目录将 `images` 替换为 `labels` 推导。
- 任何关键路径无效会输出醒目告警，并跳过该数据集处理。

### 输出目录命名

- 数据集前缀：`[数据集目录名前10字符].[后10字符]`（长度不超过 20 时直接使用全名）
- 单标签 / 多标签非合并：
  - `[dataset_short]_[label_name]_[timestamp]`
- 多标签且 `--merge true`：
  - `[dataset_short]_merge_[timestamp]`

### 输出结构

每个输出目录下会按 split 生成：

```text
<run_dir>/
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
```

### 标签过滤行为

- 读取每个 `*.txt` 标签文件，仅保留首列类别 id 属于目标类别的行。
- 原始标签文件不会被修改。
- 非合并模式下，每个输出目录只包含该类别行。
- 合并模式下，一个标签文件可包含多个目标类别行。

### 使用示例

按类别名，多类别分别输出：

```bash
python3 filter_yolo_roboflow.py \
  --data-yaml /home/ieds/datasets/helmet-det/A-ConstructionSiteSafety.v30-raw-images_latestversion.yolo26/ORIGIN/data.yaml \
  --target-names Excavator,Gloves \
  --output-root /home/ieds/Pictures/test123 \
  --merge false
```

按类别名，多类别合并输出（推荐先 dry-run）：

```bash
python3 filter_yolo_roboflow.py \
  --data-yaml /home/ieds/datasets/helmet-det/A-ConstructionSiteSafety.v30-raw-images_latestversion.yolo26/ORIGIN/data.yaml \
  --target-names Excavator,Gloves \
  --output-root /home/ieds/Pictures/test123 \
  --merge true \
  --dry-run \
  --debug
```

## Roboflow Wrapper（配置驱动）

新增包装脚本：`filter_yolo_roboflow_wrap.py`

- 默认配置文件：`configs/filter_yolo_roboflow_profiles.toml`
- 调用方式：`配置文件 + 配置名`
- 校验失败时不会调用 `filter_yolo_roboflow.py`

### Wrapper 参数

- `--config-name <name>`（必填）
- `--config-file <path>`（可选，相对/绝对路径均支持）
- `--print-command`（可选，打印最终调用命令）

### 配置示例

```toml
[profiles.demo_by_names_split]
data_yaml = "/home/ieds/datasets/helmet-det/A-ConstructionSiteSafety.v30-raw-images_latestversion.yolo26/ORIGIN/data.yaml"
target_names = "Excavator,Gloves"
output_root = "/home/ieds/Pictures/test123"
merge = "false"
dry_run = true
debug = true
```

### Wrapper 使用示例

```bash
python3 filter_yolo_roboflow_wrap.py --config-name demo_by_names_split
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
