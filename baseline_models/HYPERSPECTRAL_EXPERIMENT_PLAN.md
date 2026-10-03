# 高光谱树冠实例分割：降维与基线实验计划

## 1. 目标与实验问题

本计划比较高光谱影像的三通道表达、DSM 高程信息，以及不同检测/实例分割基线对树冠分割的影响。当前 Mask R-CNN 的 RGB + 后处理 + TTA 是强基准；新实验应先比较输入与模型本身，再单独比较后处理，避免将改进归因混在一起。

核心问题：

1. 高光谱降维得到的三通道表示是否优于现有 RGB？
2. PCA、其他无监督投影和光谱波段组合之间有何差异？
3. DSM 是否带来稳定增益？
4. 在相同输入和划分下，Mask R-CNN、Faster R-CNN、Mask2Former 等基线表现如何？Faster R-CNN 只输出框，不能作为实例掩膜指标的直接替代，需要将其与 SAM 的提示式结果单列比较。
5. TTA 与掩膜后处理的增益有多大？

## 2. 当前仓库与数据现状

- 统一数据根目录：`/home/huiying/Documents/ITCsegmentation/Data/DatasetNew/processed/`。
- 原始 HSI 产品：`balsam_hsi_1024_stride256/`，包含 `train|val|test/hsi/*.tif`、DSM 和 `annotations/{train,val,test}.json`。
- 已有三通道 PCA：`pca_alllabeled_1024_stride256_png/`；另有 `pca_1024_stride256_png/`、`pca_512_stride256_png/`。注意后二者未必与 all-labeled 使用相同标注版本或影像范围，须通过清单核验后才能横向比较。
- RGB 对照：`rgb_alllabeled_1024_stride256_png/`。
- COCO JSON 中每张影像包含 `id,file_name,dsm_path,width,height,site,split,source_row,source_col`；标注包含 `image_id,category_id,segmentation,bbox,area,iscrowd,source_id`。示例类别为 `tree`，类别 id 为 1。
- `maskrcnn/configs/config_pca_alllabeled_rgb.yaml` 已指向 PCA 三通道 PNG 和 COCO train/val/test；RGB 与 RGB+DSM 也有配置。`train_coco_rgb.py` / `train_coco_rgb_dsm.py` 是与现有 COCO PNG 标注相匹配的训练入口。
- 通用 `maskrcnn/dataset.py` 读取 GeoDataset 格式，RGB 通常读前三波段；不能直接假设它会正确处理任意 HSI TIFF 或把 PCA 参数应用到测试数据。对于现有 PNG + COCO JSON，优先沿用 `train_coco_rgb*.py` 系列，补齐统一的数据适配入口，再接其他模型。
- `maskrcnn/infer_coco_rgb_tta_nms_pixel.py` 面向 4 通道 RGB+DSM checkpoint；RGB/PCA 3 通道应使用对应 3 通道模型和推理代码，不能直接加载 4 通道权重。

执行前记录仓库版本、配置、依赖环境、原始数据清单及其校验和。数据路径在不同机器上需改成实际挂载路径。

## 3. 实验设计与控制变量

### 3.1 输入组

至少做以下组别，前三组为主实验：

| 组别 | 影像输入 | DSM | 目的 |
|---|---|---:|---|
| A | 现有 RGB 3 通道 | 否 | 当前 RGB 基准 |
| B | HSI 拟合 PCA 的 PC1-PC3 | 否 | 主 PCA 对照 |
| C | PCA 三通道 + DSM 第 4 通道 | 是 | PCA 与地形增益 |
| D | 现有 PCA all-labeled 三通道 | 否 | 复现已整理的数据产品 |
| E | PCA all-labeled + DSM | 是 | 已整理 PCA 数据的 DSM 消融 |
| F | 三个预先指定的原始光谱波段 | 否 | 简单波段选择基准 |
| G | MNF 或波段筛选后的前三通道（可选） | 否/是 | 扩展降维比较 |

`B/C` 的 PCA 应从原始 HSI 重新生成，才能严格控制拟合规则。`D/E` 用于复现当前产品；若不知道该产品 PCA 的拟合范围、缩放和波段规则，应标记为“历史数据产品复现”，不要与严格无泄漏 PCA 混为同一结论。

降维建议分两阶段：第一阶段只做 RGB、PCA-PC1:3、光谱波段组三组；第二阶段依据验证集结果扩展 MNF、波段选择或其它投影。PC 次序固定为按训练集解释方差从高到低。

### 3.2 模型组

1. **Mask R-CNN**：主分割基线，优先覆盖 A–F 及 DSM 消融。
2. **Mask2Former**：作为另一种实例/全景分割架构；其数据适配及类别映射需要统一。
3. **Faster R-CNN**：框检测基线，只报告 bbox AP；如果仓库流程接 SAM，则另报 `Faster R-CNN + SAM` 的 mask AP，单独命名，不称 Faster R-CNN 原生掩膜结果。

先做模型 × 输入组的核心矩阵（至少 Mask R-CNN 的 A–F；其他基线做 A、B、C、D 中资源允许的代表组）。不要在所有输入和模型上同时网格搜索后处理参数。

### 3.3 公平比较约束

- 同一组实验必须使用完全相同的 train/val/test 图像 ID、标注、tile 大小、stride 和无效区域处理。
- PCA、标准化、波段筛选只在训练 split 拟合；验证/测试仅 transform。禁止把 test 光谱用于拟合 PCA 或选超参数。对重叠 tile，若 tile 来自同一原始场景，划分要按源影像/区域分组，避免重叠或邻近 tile 跨 split 泄漏；沿用现有划分时先审计 `site/source_row/source_col` 与原始影像关系。
- 同一模型的 RGB/PCA 预训练初始化策略一致。若用 ImageNet 预训练，记录三通道语义差异；不得为 PCA 组偷偷更换初始化。四通道 DSM 的首层扩展初始化方式固定并记录（例如前三通道复制预训练权重，第 4 通道置零）。
- 固定优化器、学习率、epoch 上限、增强、batch size、阈值与验证频率；如显存迫使改变 batch size，记录并尽量用梯度累积保持有效 batch size。
- 至少跑 3 个种子（建议 0、1、2）；报告每 seed 结果、均值和标准差。早停/选 checkpoint 只能看 val，最终 test 仅在设置冻结后评测一次。
- 主结果先评估原始模型输出（raw）。随后在验证集上调阈值/后处理；锁定一组规则后，在 val 和 test 都跑 raw 与 postprocessed。TTA 也作为独立消融（无 TTA、有 TTA；DSM 输入的所有视图必须同步变换）。

## 4. 三通道降维数据的生成规范

建议新增独立预处理脚本（例如 `baseline_models/preprocess_hsi_to_3ch.py`），不把 PCA 逻辑散落复制到各模型的数据集类中。脚本需要支持 `pca`、`bands`，后续再扩展 `mnf`。

### 输入

- HSI COCO JSON：分别为 train、val、test，或统一 COCO JSON 中可按 `split` 筛选。
- 每个 JSON `images[].file_name` 指向原始 HSI TIFF；`dsm_path` 指向 DSM；影像及标注的 `id`、尺寸、分割掩膜保持不变。
- YAML/CLI 配置：方法、输入波段索引（明确 0-based 或 1-based）、拟合样本抽样策略、输出缩放、nodata/无效值处理、随机种子、输出目录。

### 处理操作

1. 校验 HSI/DSM 的宽高、波段数、数据类型、nodata 和 split 清单；保存错误清单并停止处理有错影像。
2. 只读 train HSI，按固定抽样策略取得像素光谱；过滤 nodata/非有限值。可按有效像素随机抽样以控制内存，抽样种子和像素数写入元数据。
3. 在 train 像素上拟合 PCA（或其他变换）；保存均值、组件、解释方差、波段顺序、输入 dtype、nodata 规则、拟合样本范围、代码版本和参数到 `transform.json` / `transform.npz`。
4. 用同一变换处理 train/val/test；将前三个分量映射到 8-bit 时，缩放上下界只能由训练集确定，并固定逐通道或全通道策略。建议优先保存 float32 多通道 TIFF（保留数值信息）；若模型入口只读 PNG，再从 float32 结果导出 PNG，同时记录每通道 `min/max` 或分位截断策略。不得每张图单独 min-max，否则同一光谱值会跨图含义漂移。
5. 输出每张影像恰好 3 个通道、尺寸不变、无 NaN/Inf；DSM 作为独立对齐文件保留，模型数据加载时拼接为第 4 通道。
6. 为每种输入生成独立 COCO JSON 副本：只改 `file_name`（及如有需要 `dsm_path`），不重编号 image/annotation、不修改 segmentation；保留原始标注数据并比较变换前后的 image id、annotation id 和 mask 面积。
7. 生成数据清单与质量检查报告：split 数量、波段数、尺寸、范围、无效像素比例、影像缺失、ID/标注匹配情况、重叠 tile 风险、transform 参数哈希。

### 输出目录建议

```text
processed/
  hsi_pca_trainfit_v1/
    train/{image,dsm}/...     # 3-channel image; DSM unchanged/aligned
    val/{image,dsm}/...
    test/{image,dsm}/...
    annotations/{train,val,test}.json
    transform.json
    manifest.csv
    quality_report.json
```

每个输入版本用独立且带版本号的目录；切勿覆盖现有 `pca_alllabeled_1024_stride256_png`。

## 5. 按顺序执行的实验操作

### 第 0 步：冻结当前强基准

**操作：** 找到并复跑当前 Mask R-CNN RGB + 后处理 + TTA 的确切 checkpoint、训练配置、数据 split 和评估阈值。分别保存 raw、postprocess、TTA 结果。记录 checkpoint SHA256。

**输入：** 当前 RGB COCO JSON、RGB/DSM（按当前模型通道）、基准 YAML、现有训练权重。

**输出：** `baseline_record.json`、训练配置快照、checkpoint、val/test 的 COCO bbox/segm 指标、预测 JSON、后处理参数及 TTA 设置。若旧结果无法复现，先将当前最好可复现设置定义为基准，不以日志口述数字做比较。

### 第 1 步：核验数据划分和标注

**操作：** 核对 RGB、PCA、原始 HSI 三种 COCO 文件中 split 数量、image ID、源场景坐标、标注数量、空标注比例、影像尺寸和 DSM 对齐。按源图/区域审查 split 泄漏与 stride 256 的重叠 tile 是否跨 split。确认 all-labeled 的标注定义在 RGB/PCA/HSI 一致。

**输入：** 三个数据产品的 `annotations/{train,val,test}.json` 和对应文件。

**输出：** `data_audit.json`、每 split 的 `manifest.csv`、异常列表；如 RGB/PCA 对应图像不完全一致，生成共有 image ID 的实验子集，或单独作为不同数据版本，不直接比较分数。

### 第 2 步：建立严格的降维产品

**操作：** 从原始 HSI 生成 train-only PCA 三通道；同步生成固定原始波段三通道对照。先完成 PCA 与波段组，不要先实现太多降维算法。按第 4 节规格保存影像、DSM 路径、COCO JSON、拟合参数和质量报告。

**输入：** HSI `train/val/test` COCO JSON、原始 HSI TIFF、预先注册的波段索引和预处理配置。

**输出：** 版本化三通道数据目录、三份 COCO JSON、`transform.json`/`transform.npz`、manifest 与质量报告。

### 第 3 步：适配并做数据加载冒烟检查

**操作：** 统一模型数据适配，使 3 通道 RGB/PCA/bands 和 4 通道 image+DSM 使用相同 COCO 标注解码、归一化策略、增强同步和数据 ID。检查空实例图、边界框范围、掩膜面积、DSM 对齐、通道数。只做数据检查，不耗费完整训练。

**输入：** 各输入组的一张 train/val/test 样本及 COCO JSON。

**输出：** `loader_check.json`，至少包含 tensor shape/dtype/range、图像 ID、实例数、框与 mask 一致性、通道值范围；保存 RGB/PCA/DSM/标注叠加的抽样可视化 PNG。

### 第 4 步：跑 Mask R-CNN 输入消融矩阵

**操作：** 先用 seed 0 验证训练配置和资源，再跑 3 seeds。核心组为 A RGB、B 新 PCA、C 新 PCA+DSM、D 现有 PCA、E 现有 PCA+DSM、F bands。每个 run 都以验证集 segm AP 选择 best checkpoint。若计算资源有限，可先对全部组跑 seed 0 做筛选，但入选结论组须补齐 3 seeds。

**输入：** 第 2 步各组 COCO JSON；固定 Mask R-CNN 训练代码、相同训练超参及 seed 列表。每组配置分别写入 `maskrcnn/configs/`，明确 `ann_train/val/test`、`input_channels`、DSM 归一化、输出目录和种子。

**输出：** 每 run 独立目录：配置快照、epoch 日志、best/周期 checkpoint、val 指标；结果表至少含 `input,DSM,seed,best_epoch,val_bbox_AP,val_bbox_AP50,val_segm_AP,val_segm_AP50`。

示意命令（按配置字段和机器环境调整）：

```bash
cd baseline_models/maskrcnn
python train_coco_rgb.py configs/config_pca_alllabeled_rgb.yaml
python train_coco_rgb_dsm.py configs/config_rgb_alllabeled_rgb_dsm.yaml
```

现有配置中的 `save_ckpt_path`、数据路径必须对应本次组别；训练入口中调用的 builder、通道数和评估脚本也必须匹配。新 PCA / bands 组应建立独立配置，不覆盖历史配置。

### 第 5 步：比较其他 baseline

**操作：** 将 Mask R-CNN 选出的代表输入（RGB、严格 PCA、严格 PCA+DSM）扩展到 Mask2Former；Faster R-CNN 使用同一三通道影像和 COCO 标注跑 bbox 指标，若加入 SAM 则保存提示与 SAM 版本并独立评估 mask 指标。先保证数据适配与指标口径，再扩大输入组。

**输入：** 相同 train/val/test ID 的三通道/四通道数据和标签；各模型明确的预训练权重、通道适配和训练配置。

**输出：** 每模型每组独立 checkpoint、配置和预测；同一汇总表中注明 `task=bbox`、`task=instance_mask` 或 `task=detector+SAM`，不可把 bbox AP 与 mask AP 混作同一个任务指标。

### 第 6 步：后处理和 TTA 消融

**操作：** 仅对第 4/5 步的代表模型，在 val 上选 mask score、mask threshold、NMS IoU、containment 等参数。固定参数后各跑 raw、postprocess、TTA、TTA+postprocess；DSM 输入所有旋转/翻转需一起应用到 DSM。不要使用 test 选择参数。

**输入：** 冻结 checkpoint、val/test COCO JSON、候选后处理范围、TTA 变换定义。

**输出：** 参数配置、各变体预测 COCO JSON、每图实例数、bbox/segm AP 与 AP50、可视化样例；最终 test 结果只在设置冻结后生成。

### 第 7 步：最终统计与归档

**操作：** 汇总各 seed 均值 ± 标准差；列出最佳 epoch、参数量、推理时间/图、显存（如可测）、硬件与软件环境。对主要差异做配对图像级 bootstrap 置信区间（按源场景聚类更稳妥），避免只凭单 seed 小数点差距下结论。

**输入：** 所有训练日志、配置快照、checkpoint 元数据、val/test 预测与指标。

**输出：** `results.csv`、`results_summary.md`、最终配置和模型文件清单、随机种子、数据/模型哈希、失败 run 记录、论文/报告可用表格与示例可视化。

## 6. 建议输出格式

### 训练运行目录

```text
Output/hs_baselines/<dataset_version>/<model>/<input_group>/seed_<seed>/
  config.yaml
  train.log
  best_model.pt
  metrics.json
  predictions_val.json
  predictions_test.json       # 仅最终冻结设置运行
  settings.json               # checkpoint/data/transform 哈希、阈值、软件版本
  visualizations/
```

### 汇总 CSV 字段

```text
run_id,dataset_version,split_hash,model,task,input_method,input_channels,dsm,
pretrained,seed,best_epoch,val_bbox_AP,val_bbox_AP50,val_segm_AP,val_segm_AP50,
test_bbox_AP,test_bbox_AP50,test_segm_AP,test_segm_AP50,postprocess,tta,
inference_ms_per_image,checkpoint_sha256,transform_sha256,config_path
```

未运行的指标留空，不填 0。CSV 每行对应一个 seed 和一个推理变体；另提供按组聚合的 mean/std 表。

## 7. 指标与结果解读

- 实例掩膜主指标：COCO `segm AP@[.50:.95]`、`segm AP50`、`segm AP75`；同步报告 bbox AP 便于对比检测质量。
- 辅助诊断：precision/recall 或 F1（必须声明匹配 IoU 与 score threshold）、每图实例数、重复掩膜/重叠像素比例。
- Mask R-CNN 与 Mask2Former 的 raw 指标作为模型基线；后处理和 TTA 结果作为独立列。
- 先比较同一模型内部输入组的变化，再比较同一输入组内模型变化。DSM 增益按配对组计算：`(同方法 + DSM) - (同方法无 DSM)`。
- 解释方差、组件图像和光谱载荷仅用于描述预处理，不等同于树冠任务有效性；最终按 val 选择方案，test 用于无偏最终报告。

## 8. 开始前需补齐/确认的实现项

1. 查明 `pca_alllabeled` 的 PCA 拟合范围、每通道缩放方法、HSI 波段清单及无效值规则；若无法追溯，将其标为既有产品，并生成 train-only PCA 作为严格实验组。
2. 检查 RGB 与 PCA COCO 中图像 ID、标注和 DSM 一一对应；样例目前显示都有 `dsm_path`，但路径及影像网格仍需全量校验。
3. 确认 Mask R-CNN 3/4 通道训练与 TTA 推理入口匹配；当前 TTA 文件默认是 RGB+DSM 4 通道，不适用于 3 通道 PCA。
4. 将模型训练/评估脚本参数化，确保三通道源可以是 RGB、PCA 或指定 band，不让数据集类根据目录名猜输入类型。
5. 对齐 Faster R-CNN、Mask2Former 的 split、类别编号、影像归一化、DSM 预处理、COCO 指标和后处理定义。
6. 审核现有 `pca_1024`、`pca_512` 与 all-labeled 产品的标签覆盖范围及划分定义。特别是 512/1024 tile 的尺度差异会改变目标像素尺寸，不能当成单纯降维差异。

## 9. 推荐首轮最小可执行批次

为尽快得到可信答案，建议先做：

1. 数据审计与当前 RGB 基准复现。
2. 生成严格 train-only PCA 三通道和 band 三通道，使用与 RGB 完全相同的图像 ID / 标注。
3. Mask R-CNN：RGB、PCA、PCA+DSM、bands 四组，各 3 seeds；先 raw 评估。
4. 对 val 最优输入组比较 raw、TTA、后处理、TTA+后处理。
5. 再把 RGB/PCA/PCA+DSM 扩展到 Mask2Former 与 Faster R-CNN（后者如配 SAM 则另列）。
6. 最后冻结配置并运行一次 test，归档预测、指标、哈希和可视化。

这套顺序先回答降维与 DSM 是否有用，再投资扩展架构和推理策略；每个阶段都有明确输入、输出与继续条件。
