# 高校图书馆个性化图书推荐算法

本项目实现了一个面向高校图书馆真实图书借阅场景(脱敏)的推荐算法，主要包含 ALS、LightGCN 召回，神经网络排序。


## 数据文件

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
数据集这里暂时不提供如果需要的朋友可以联系我~

