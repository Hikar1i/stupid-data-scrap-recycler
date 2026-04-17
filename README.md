# YOLO Dataset Filter & Conversion Toolkit

该项目包含以下脚本，可单独使用，也可通过 wrapper 串联为一键式 pipeline：

| 脚本 | 功能 |
|---|---|
| `filter_coco_category.py` | 从 COCO JSON 按类别筛选图像和标签 |
| `filter_yolo_roboflow.py` | 从 Roboflow YOLO 数据集按类别筛选 |
| `remap_yolo_labels.py` | YOLO 标签序号重映射 |
| `yolo_to_labelme.py` | YOLO 标签转 labelme JSON 格式 |
| `filter_coco_category_wrap.py` | COCO filter wrapper，支持 pipeline |
| `filter_yolo_roboflow_wrap.py` | Roboflow filter wrapper，支持 pipeline |
| `gen_roboflow_filter_config.py` | 批量扫描 Roboflow 数据集目录，按关键词自动生成过滤配置文件 |
| `batch_run_profiles.py` | 批量执行 wrapper 的所有配置段 |

---

# filter_coco_category.py — COCO 类别过滤

该脚本从 COCO 标注中按类别筛选图像，并将图像与对应标签拷贝到目标目录。

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
- `filter_coco_category.py`、`filter_yolo_roboflow.py`、`remap_yolo_labels.py` 仅依赖标准库
- `yolo_to_labelme.py` 需要 **Pillow**（用于读取图像尺寸）：`pip install Pillow`
- wrapper 脚本读取 TOML 时优先使用 Python 3.11+ 内置 `tomllib`；低版本请安装 `tomli`：`pip install tomli`

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

默认文件 `configs/filter_coco_category_profiles.toml` 中可写多组配置。

> **注意**：配置文件顶部需包含 `[meta]` 节声明 wrapper 类型，供 `batch_run_profiles.py` 校验使用；各 wrapper 脚本本身会忽略该节。

```toml
[meta]
wrapper_type = "coco_category"

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

# filter_yolo_roboflow.py — Roboflow YOLO 过滤脚本

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

- 数据集前缀：`[数据集目录名前20字符].[后20字符]`（长度不超过 40 时直接使用全名）
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

> **注意**：配置文件顶部需包含 `[meta]` 节声明 wrapper 类型，供 `batch_run_profiles.py` 校验使用；各 wrapper 脚本本身会忽略该节。

```toml
[meta]
wrapper_type = "yolo_roboflow"

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

---

# remap_yolo_labels.py — YOLO 标签序号重映射

将 YOLO 标签文件中的类别 ID 替换为新的 ID（如 `11→0, 8→1`）。

## 参数

- `--source-dir <path>`（必填）：YOLO 数据集根目录或单个 split 目录。脚本递归查找所有 `labels/` 子目录。
- `--mapping <src:dst,...>`（必填）：映射关系，如 `11:0,8:1`。
- `--inplace true|false`（默认 `false`）：是否直接修改原始标签文件。`true` 时 `--output-dir` 无效。
- `--output-dir <path>`（可选）：输出目录。未指定时，在每个 `labels/` 的同级目录创建 `labels_remapping_<timestamp>/`。
- `--dry-run`：预览，不写入任何文件。
- `--debug`：打印每行替换/跳过详情。

## 冲突检测

当目标 ID 已存在于同一文件的其他行时，该行替换被跳过（其他行正常替换）。运行结束后会以醒目格式打印冲突统计，**提示用户检查 mapping 是否正确后重新运行**。

## 使用示例

```bash
python3 remap_yolo_labels.py \
  --source-dir /path/to/yolo/dataset \
  --mapping "11:0,8:1" \
  --dry-run --debug
```

---

# yolo_to_labelme.py — YOLO 转 labelme JSON

将 YOLO 格式 `*.txt` 标签转换为 labelme 的 `*.json` 格式（矩形框，4 角点坐标）。

## 参数

- `--source-dir <path>`（必填）：支持以下三种形式：
  - YOLO 数据集根目录（含 `train/valid/test` 子目录）；
  - 单个 split 目录（直接含 `images/` 和 `labels/`）；
  - pipeline reindex 阶段输出的 `labels_remapping_<timestamp>/` 目录（脚本自动检测同级 `images/`）。
- `--mapping <id:name,...>` / `--classes-file <path>`（二选一，必填）：
  - `--mapping`：内联映射，支持含空格的类别名，如 `0:cat,6:No helmet,7:Safety Vest`。
  - `--classes-file`：txt 文件，第一行 = ID 0，逐行对应类别名（与 classes.txt 格式一致）。
- `--output-dir <path>`（可选）：未指定时 JSON 写入 `images/` 目录（与图像共存）；指定时按源目录结构输出到 `<output_dir>/<split>/labelme_<timestamp>/`。
- `--overwrite true|false`（默认 `false`）：已存在的 JSON 是否覆盖。
- `--dry-run`：预览，不创建文件。
- `--debug`：打印每个文件的转换路径。

## 转换规则

- YOLO `cx cy w h`（归一化）→ labelme 矩形 4 角点（像素坐标）：左上→右上→右下→左下。
- 图像宽高通过 Pillow 读取（**需安装 Pillow**）。
- 标签中未在 mapping 中定义的 ID 转换为 `"unknown"`，并以醒目警告输出。
- 转换完成后，自动在**数据集根目录**下生成 `classes.txt`（每行一个类别名，行号即 ID，从 0 开始）。若文件已存在且未指定 `--overwrite true` 则跳过。

## 使用示例

```bash
# 使用 classes.txt
python3 yolo_to_labelme.py \
  --source-dir /path/to/yolo/dataset \
  --classes-file /path/to/classes.txt \
  --output-dir /path/to/labelme_output

# 使用内联 mapping（含空格类别名）
python3 yolo_to_labelme.py \
  --source-dir /path/to/yolo/dataset \
  --mapping "0:cat,6:No helmet,7:Safety Vest" \
  --dry-run
```

---

# Pipeline — 一键过滤→重设序号→转 labelme

两个 wrapper 脚本（`filter_coco_category_wrap.py` 和 `filter_yolo_roboflow_wrap.py`）均支持 pipeline 模式，可将三个阶段串联：

```
filter → remap_yolo_labels → yolo_to_labelme
```

## 配置格式（嵌套 TOML）

通过在 profile 中添加 `[profiles.<name>.reindex]` 和/或 `[profiles.<name>.cvtlabelme]` 子节来开启对应阶段。**子节存在即开启，缺失即跳过**。

```toml
[profiles.my_pipeline.filter]
category_names = "boom,fence"
json_dir = "/path/to/coco/jsons"
label_root = "/path/to/labels/yolo"
output_root = "/path/to/output"
merge = "true"

[profiles.my_pipeline.reindex]
mapping = "11:0,8:1"
inplace = false

[profiles.my_pipeline.cvtlabelme]
mapping = "0:boom,1:fence"
overwrite = false
```

## 各阶段配置键

### `[...filter]` 键
与对应 filter 脚本参数一致（见各 filter 脚本参数说明）。

### `[...reindex]` 键
| 键 | 说明 |
|---|---|
| `mapping` | 必填，`src:dst,...` 格式 |
| `inplace` | `true/false`，默认 `false` |
| `output_dir` | 可选，reindex 输出目录 |
| `dry_run` | 布尔 |
| `debug` | 布尔 |

### `[...cvtlabelme]` 键
| 键 | 说明 |
|---|---|
| `mapping` | 与 `classes_file` 二选一，`id:name,...` 格式 |
| `classes_file` | classes.txt 路径 |
| `output_dir` | 可选，JSON 输出目录 |
| `overwrite` | `true/false`，默认 `false` |
| `dry_run` | 布尔 |
| `debug` | 布尔 |

## Pipeline 阶段链接原理

- filter 脚本在 pipeline 模式下会打印 `OUTPUT_DIR:<path>` 行（通过 `--print-output-dir` 标志触发）。
- wrapper 解析这些路径后，将其作为 reindex 的 `--source-dir` 输入。
- 若 reindex 的 `inplace=false`，reindex 同样打印 `OUTPUT_DIR:<path>`，传给 cvtlabelme；若 `inplace=true`，则 cvtlabelme 直接使用 filter 输出目录。
- filter 产生多个输出目录（`merge=false` + 多类别）时，reindex/cvtlabelme 对每个目录独立运行。

## ⚠️ 边界情况：仅配置 cvtlabelme 无 filter

若 wrapper profile 中无 `[...filter]` 节，但有 `[...cvtlabelme]` 节，wrapper 无法自动推断源目录，cvtlabelme 阶段的 `source_dir` 需在 `[...cvtlabelme]` 节中显式提供：

```toml
[profiles.cvtlabelme_only.cvtlabelme]
source_dir = "/path/to/existing/yolo/dataset"
mapping = "0:cat,1:dog"
```

同时 wrapper 会打印醒目警告，提示用户 YOLO 标签 ID 未经 reindex 阶段处理，可能大量输出 `unknown` 类别。

## Pipeline 使用示例

```bash
# COCO filter + reindex + cvtlabelme
python3 filter_coco_category_wrap.py --config-name my_pipeline

# Roboflow filter + reindex only
python3 filter_yolo_roboflow_wrap.py --config-name my_pipeline --print-command
```

旧格式（仅含 filter 参数的扁平 profile）不受影响，向后兼容。

---

# gen_roboflow_filter_config.py — 批量生成过滤配置文件

扫描指定目录下所有 Roboflow 数据集子目录，读取每个数据集的 `data.yaml`，按关键词匹配类别后，自动生成可供 `filter_yolo_roboflow_wrap.py` 执行的 TOML 配置文件。

## 功能用途

- 自动扫描目录下一层含 `data.yaml` 的子目录（Roboflow 标准结构）。
- 以 `*关键词*`（子串）方式匹配 `data.yaml` 中 `names` 数组的类别名，支持多关键词（逗号分隔，OR 逻辑）。
- 为每个命中的数据集生成一段完整 pipeline profile（filter → reindex → cvtlabelme）。
- 通过配置模板（`configs/_template/`）控制非动态字段的默认值及启用哪些流水线阶段。
- 动态字段由脚本自动填充，无需手动配置：
  - `filter.data_yaml`：扫描到的数据集路径
  - `filter.target_names`：匹配到的类别名列表
  - `reindex.mapping`：`原始序号:新序号,...`（如 `3:0,6:1`）
  - `cvtlabelme.mapping`：`新序号:类别名,...`（如 `0:crane,1:tower crane`）

## 参数说明

### 必选参数

- `--scan-dir <DIR>`：扫描目录，该目录下一层子目录中有 `data.yaml` 的均视为 Roboflow 数据集。
- `--output-config <FILE>`：输出配置文件路径。仅文件名（不含路径分隔符）时自动放入 `configs/` 目录；含路径分隔符时按绝对/相对路径处理。
- `--keywords <KW[,KW2,...]>`：类别名匹配关键词，多个关键词用英文逗号分隔（任一命中即算匹配）。

### 可选参数

- `--overwrite`：若目标配置文件已存在则覆盖，默认不覆盖。
- `--case-sensitive`：匹配时区分大小写，默认不区分。
- `--filter-output-root <DIR>`：写入生成配置中的过滤结果输出目录（即 `filter_yolo_roboflow.py` 的 `--output-root`）。与模板中 `[filter].output_root` 必须存在其一，CLI 参数优先。
- `--template <FILE>`：配置模板文件名（自动在 `configs/_template/` 下查找）或绝对/相对路径。未指定时使用内置默认值并生成完整三阶段配置。

## 配置模板

模板文件存放于 `configs/_template/` 目录，TOML 格式。动态字段无需在模板中配置，脚本自动生成。

**模板控制哪些流水线阶段被生成**：
- 模板中存在 `[reindex]` 节 → 生成的配置包含 reindex 阶段
- 模板中存在 `[cvtlabelme]` 节 → 生成的配置包含 cvtlabelme 阶段
- 未指定模板时，默认生成全三阶段配置

内置示例模板：`configs/_template/default_pipeline.toml`

```toml
[filter]
output_root = ""    # 也可通过 --output-root 指定，命令行优先
merge = true
dry_run = false
debug = false

[reindex]
inplace = true

[cvtlabelme]
overwrite = false
```

## 使用示例

### 1) 基础用法（使用内置默认值，生成完整三阶段配置）

```bash
python3 gen_roboflow_filter_config.py \
  --scan-dir /path/to/roboflow_datasets \
  --output-config crane_filter.toml \
  --keywords "crane" \
  --filter-output-root /path/to/output
```

### 2) 多关键词匹配（OR 逻辑）

```bash
python3 gen_roboflow_filter_config.py \
  --scan-dir /path/to/roboflow_datasets \
  --output-config crane_filter.toml \
  --keywords "crane,tower crane,boom" \
  --filter-output-root /path/to/output
```

### 3) 使用模板文件（output_root 在模板中配置）

```bash
python3 gen_roboflow_filter_config.py \
  --scan-dir /path/to/roboflow_datasets \
  --output-config crane_filter.toml \
  --keywords "crane" \
  --template default_pipeline.toml
```

### 4) 覆盖已有配置文件，区分大小写匹配

```bash
python3 gen_roboflow_filter_config.py \
  --scan-dir /path/to/roboflow_datasets \
  --output-config /abs/path/to/output.toml \
  --keywords "Crane,Tower Crane" \
  --filter-output-root /path/to/output \
  --case-sensitive \
  --overwrite
```

## 生成的配置文件结构示例

```toml
# 自动生成配置文件
# 生成时间：2025-04-17 14:00:00
# 扫描目录：/path/to/roboflow_datasets
# 匹配关键词：crane

[meta]
wrapper_type = "yolo_roboflow"

[profiles.20250217_v2i_yolov11.filter]
data_yaml = "/path/to/roboflow_datasets/-----20250217.v2i.yolov11/data.yaml"
target_names = "crane,tower crane"
output_root = "/path/to/output"
merge = true
dry_run = false
debug = false

[profiles.20250217_v2i_yolov11.reindex]
mapping = "3:0,6:1"
inplace = true

[profiles.20250217_v2i_yolov11.cvtlabelme]
mapping = "0:crane,1:tower crane"
overwrite = false
```

生成后可直接通过 `filter_yolo_roboflow_wrap.py` 或 `batch_run_profiles.py` 执行：

```bash
# 单个 profile 执行
python3 filter_yolo_roboflow_wrap.py \
  --config-file configs/crane_filter.toml \
  --config-name 20250217_v2i_yolov11

# 批量执行所有 profile
python3 batch_run_profiles.py \
  --wrapper filter_yolo_roboflow_wrap.py \
  --config-file configs/crane_filter.toml
```

---

# batch_run_profiles.py — 批量执行所有配置段

对 TOML 配置文件中所有有效 profile，逐一调用对应 wrapper 脚本执行，并汇总结果。

## 功能用途

- 读取 TOML 配置文件，列出所有有效/无效配置段（TOML 语法层面）。
- 在执行前校验 wrapper 脚本类型与配置文件声明类型（`[meta].wrapper_type`）是否匹配，不匹配则拒绝执行。
- 逐一以子进程方式调用 wrapper，实时流式输出，不引入任何 wrapper 内部依赖（低耦合）。
- 全部执行完后打印成功/失败汇总；支持 `--fail-fast` 遇错立即终止。

## 配置文件 `[meta]` 节

`batch_run_profiles.py` 通过 TOML 顶部的 `[meta]` 节判断配置文件类型：

| `wrapper_type` 值 | 对应 wrapper 脚本 |
|---|---|
| `yolo_roboflow` | `filter_yolo_roboflow_wrap.py` |
| `coco_category` | `filter_coco_category_wrap.py` |

各 wrapper 脚本本身不读取 `[meta]` 节，向下兼容已有配置文件（只需在文件顶部新增两行）。

## 参数说明

| 参数 | 是否必填 | 说明 |
|---|---|---|
| `--wrapper` | 必填 | wrapper 脚本文件名或路径，如 `filter_yolo_roboflow_wrap.py` |
| `--config-file` | 必填 | TOML 配置文件路径（绝对或相对） |
| `--dry-run` | 可选 | 仅列出配置段，不实际执行任何命令 |
| `--print-command` | 可选 | 透传给各 wrapper 子进程，执行前打印最终命令 |
| `--fail-fast` | 可选 | 遇到首个失败配置段时立即终止（默认：继续执行全部） |

## 使用示例

### 1) 批量执行所有配置段（预览模式）

```bash
python3 batch_run_profiles.py \
  --wrapper filter_yolo_roboflow_wrap.py \
  --config-file configs/filter_yolo_roboflow_profiles.toml \
  --dry-run
```

### 2) 正式批量执行，遇错继续（默认行为）

```bash
python3 batch_run_profiles.py \
  --wrapper filter_yolo_roboflow_wrap.py \
  --config-file configs/filter_yolo_roboflow_profiles.toml
```

### 3) 遇到首个失败立即终止

```bash
python3 batch_run_profiles.py \
  --wrapper filter_coco_category_wrap.py \
  --config-file configs/filter_coco_category_profiles.toml \
  --fail-fast
```

### 4) 执行时同步打印各 wrapper 的最终命令

```bash
python3 batch_run_profiles.py \
  --wrapper filter_yolo_roboflow_wrap.py \
  --config-file configs/filter_yolo_roboflow_profiles.toml \
  --print-command
```

### 5) wrapper 类型不匹配时的错误示例

```bash
# 会报错并拒绝执行
python3 batch_run_profiles.py \
  --wrapper filter_coco_category_wrap.py \
  --config-file configs/filter_yolo_roboflow_profiles.toml
# → [错误] wrapper 类型不匹配！
```

