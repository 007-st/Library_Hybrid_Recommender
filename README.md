# 高校图书馆混合推荐系统（PyTorch）

本项目实现了一个面向高校图书馆借阅场景的混合推荐系统，主要包含 ALS、LightGCN、Item2Vec 多路召回，BPR 神经网络精排，图书编号序列缺口规则推荐，以及基于规则置信度的动态软融合。

## 项目结构

```text
library_hybrid_recommender/
├── configs/
│   └── default.yaml
├── data/
│   └── raw/                  # 放置 book.csv、inter.csv、user.csv
├── src/
│   ├── data/                 # 数据清洗、划分和特征处理
│   ├── evaluation/           # 推荐指标与离线评估
│   ├── inference/            # 候选生成、融合与推荐
│   ├── models/               # ALS、LightGCN、Item2Vec、精排模型
│   ├── rules/                # 序列缺口规则推荐
│   ├── training/             # 召回和精排训练流程
│   └── utils/                # 配置、日志、随机种子等工具
├── .gitignore
├── README.md
├── requirements.txt
└── run.py
```

`data/processed/`、`outputs/checkpoints/`、`outputs/reports/` 和 `outputs/recommendations/` 会在程序运行时自动创建，不包含在项目压缩包中。

## 数据文件

将以下三个 CSV 文件放入 `data/raw/`。

### book.csv

```text
book_id,题名,作者,出版社,一级分类,二级分类
```

### inter.csv

```text
inter_id,user_id,book_id,借阅时间,还书时间,续借时间,续借次数
```

### user.csv

```text
借阅人,性别,DEPT,年级,类型
```

若交互文件名称不是 `inter.csv`，请修改 `configs/default.yaml` 中的 `paths.inter_file`。

## 环境安装

推荐使用 Python 3.10 或 Python 3.11。

```bash
conda create -n library-recsys python=3.11 -y
conda activate library-recsys
pip install -r requirements.txt
```

使用 GPU 时，请先根据本机 CUDA 版本安装对应的 PyTorch，再安装其余依赖。

## 运行方法

### 一键完成预处理、召回训练、精排训练和验证集评估

```bash
python run.py --config configs/default.yaml all
```

### 分阶段运行

```bash
python run.py --config configs/default.yaml preprocess
python run.py --config configs/default.yaml train-recall
python run.py --config configs/default.yaml train-ranker
python run.py --config configs/default.yaml evaluate --split valid
python run.py --config configs/default.yaml evaluate --split test
```

### 为指定用户生成推荐

```bash
python run.py --config configs/default.yaml recommend --user-id "用户原始ID" --topk 10
```

### 导出 CPU INT8 精排模型

```bash
python run.py --config configs/default.yaml quantize
```

## 配置说明

主要参数统一保存在 `configs/default.yaml` 中，包括设备、数据路径、数据划分、ALS、LightGCN、Item2Vec、神经精排、规则置信度、融合权重和评估指标。

设备默认设置为：

```yaml
project:
  device: auto
```

程序会在 CUDA 可用时使用 GPU，否则自动使用 CPU。真实数据、预处理结果、模型参数和推荐结果均已通过 `.gitignore` 排除，不会上传到 GitHub。
