from __future__ import annotations

import argparse
from pathlib import Path

from src.data.preprocess import preprocess_dataset
from src.evaluation.evaluator import evaluate
from src.inference.hybrid import HybridRecommender
from src.inference.quantize import export_int8_ranker
from src.training.ranker_trainer import train_ranker
from src.training.recall_trainer import train_recall_models
from src.utils.config import ensure_project_dirs, load_config, project_path
from src.utils.seed import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="高校图书馆双路混合推荐系统")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML 配置文件路径")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preprocess", help="数据清洗、划分、映射及特征构建")
    subparsers.add_parser("train-recall", help="训练 ALS、LightGCN、Item2Vec")
    subparsers.add_parser("train-ranker", help="训练 BPR 神经精排模型")
    evaluate_parser = subparsers.add_parser("evaluate", help="离线评估")
    evaluate_parser.add_argument("--split", choices=["valid", "test"], default="valid")
    recommend_parser = subparsers.add_parser("recommend", help="为单个用户生成推荐")
    recommend_parser.add_argument("--user-id", required=True)
    recommend_parser.add_argument("--topk", type=int, default=10)
    subparsers.add_parser("quantize", help="导出 CPU INT8 TorchScript 精排模型")
    subparsers.add_parser("all", help="依次执行预处理、召回训练、精排训练和验证评估")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    seed_everything(int(cfg.project.seed))

    if args.command == "preprocess":
        preprocess_dataset(cfg)
    elif args.command == "train-recall":
        train_recall_models(cfg)
    elif args.command == "train-ranker":
        train_ranker(cfg)
    elif args.command == "evaluate":
        evaluate(cfg, args.split)
    elif args.command == "recommend":
        system = HybridRecommender(cfg)
        result = system.recommend(args.user_id, args.topk)
        output_path = project_path(cfg, cfg.paths.recommendation_dir) / f"user_{args.user_id}_top{args.topk}.csv"
        result.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(result.to_string(index=False))
        print(f"\n推荐结果已保存：{output_path}")
    elif args.command == "quantize":
        output = export_int8_ranker(cfg)
        print(f"INT8 精排模型已保存：{output}")
    elif args.command == "all":
        preprocess_dataset(cfg)
        train_recall_models(cfg)
        train_ranker(cfg)
        evaluate(cfg, "valid")


if __name__ == "__main__":
    main()
