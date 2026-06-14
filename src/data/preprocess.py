from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from src.data.schema import (
    BOOK_ALIASES,
    INTER_ALIASES,
    USER_ALIASES,
    standardize_frame,
)
from src.utils.config import AttrDict, project_path
from src.utils.io import save_json, save_pickle
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"未找到数据文件：{path}")
    encodings = ("utf-8-sig", "utf-8", "gb18030")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"无法识别 {path} 的编码") from last_error


def _clean_string(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().replace({"nan": "", "None": ""})


def _parse_datetime(series: pd.Series) -> pd.Series:
    raw = _clean_string(series)
    parsed = pd.to_datetime(raw, errors="coerce")
    missing = parsed.isna() & raw.ne("")
    if missing.any():
        # 兼容 20240101123000、20240101 等纯数字时间。
        digits = raw[missing].str.replace(r"\.0$", "", regex=True)
        for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d"):
            reparsed = pd.to_datetime(digits, format=fmt, errors="coerce")
            parsed.loc[missing & reparsed.notna()] = reparsed[reparsed.notna()]
            missing = parsed.isna() & raw.ne("")
            if not missing.any():
                break
    return parsed


def _iterative_kcore(
    interactions: pd.DataFrame,
    min_user: int,
    min_item: int,
    max_rounds: int,
) -> pd.DataFrame:
    frame = interactions.copy()
    for _ in range(max_rounds):
        before = len(frame)
        user_counts = frame["user_id"].value_counts()
        item_counts = frame["book_id"].value_counts()
        frame = frame[
            frame["user_id"].isin(user_counts[user_counts >= min_user].index)
            & frame["book_id"].isin(item_counts[item_counts >= min_item].index)
        ]
        if len(frame) == before:
            break
    return frame.reset_index(drop=True)


def _split_user_history(group: pd.DataFrame, strategy: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group = group.sort_values(["borrow_time", "inter_id"], kind="stable")
    if strategy == "pdf_last_day":
        last_day = group["borrow_time"].dt.normalize().max()
        if pd.isna(last_day) or group["borrow_time"].dt.normalize().nunique() <= 1:
            cut = max(1, int(math.floor(len(group) * 0.9)))
            return group.iloc[:cut], group.iloc[cut:], group.iloc[0:0]
        valid = group[group["borrow_time"].dt.normalize() == last_day]
        train = group[group["borrow_time"].dt.normalize() < last_day]
        return train, valid, group.iloc[0:0]

    # 推荐系统研究中更完整的离线验证：最后一天测试、倒数第二天验证。
    days = sorted(group["borrow_time"].dt.normalize().dropna().unique())
    if len(days) >= 3:
        valid_day, test_day = days[-2], days[-1]
        train = group[group["borrow_time"].dt.normalize() < valid_day]
        valid = group[group["borrow_time"].dt.normalize() == valid_day]
        test = group[group["borrow_time"].dt.normalize() == test_day]
        return train, valid, test
    if len(group) >= 3:
        return group.iloc[:-2], group.iloc[-2:-1], group.iloc[-1:]
    if len(group) == 2:
        return group.iloc[:1], group.iloc[0:0], group.iloc[1:]
    return group, group.iloc[0:0], group.iloc[0:0]


def _reduce_sparse(matrix: sparse.spmatrix, target_dim: int, seed: int) -> tuple[np.ndarray, object | None]:
    n_rows, n_cols = matrix.shape
    output = np.zeros((n_rows, target_dim), dtype=np.float32)
    max_dim = min(target_dim, max(0, n_rows - 1), max(0, n_cols - 1))
    if max_dim <= 0 or matrix.nnz == 0:
        return output, None
    reducer = TruncatedSVD(n_components=max_dim, random_state=seed)
    reduced = reducer.fit_transform(matrix).astype(np.float32)
    output[:, :max_dim] = reduced
    return output, reducer


def _build_user_features(users: pd.DataFrame, target_dim: int, seed: int) -> tuple[np.ndarray, dict]:
    columns = ["gender", "dept", "grade", "user_type"]
    values = users[columns].fillna("未知").astype(str)
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=1, sparse_output=True)
    encoded = encoder.fit_transform(values)
    reduced, reducer = _reduce_sparse(encoded, target_dim, seed)
    return reduced, {"encoder": encoder, "reducer": reducer, "columns": columns}


def _build_item_features(
    books: pd.DataFrame,
    target_dim: int,
    max_text_features: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    categorical_columns = ["publisher", "category1", "category2"]
    categorical = books[categorical_columns].fillna("未知").astype(str)
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True)
    categorical_matrix = encoder.fit_transform(categorical)

    text = (
        books[["title", "author", "publisher", "category1", "category2"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=2,
        max_features=max_text_features,
        sublinear_tf=True,
    )
    try:
        text_matrix = vectorizer.fit_transform(text)
    except ValueError:
        text_matrix = sparse.csr_matrix((len(books), 0), dtype=np.float32)
        vectorizer = None
    combined = sparse.hstack([categorical_matrix, text_matrix], format="csr")
    reduced, reducer = _reduce_sparse(combined, target_dim, seed)
    return reduced, {
        "categorical_encoder": encoder,
        "text_vectorizer": vectorizer,
        "reducer": reducer,
        "categorical_columns": categorical_columns,
    }


def _save_split(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        "inter_id", "user_id", "book_id", "user_idx", "item_idx",
        "borrow_time", "return_time", "renew_time", "renew_count", "weight",
    ]
    output = frame[columns].copy()
    for col in ("borrow_time", "return_time", "renew_time"):
        output[col] = output[col].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    output.to_csv(path, index=False, encoding="utf-8-sig")


def preprocess_dataset(cfg: AttrDict) -> None:
    raw_dir = project_path(cfg, cfg.paths.raw_dir)
    processed_dir = project_path(cfg, cfg.paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    book_path = raw_dir / cfg.paths.book_file
    inter_path = raw_dir / cfg.paths.inter_file
    if not inter_path.exists() and cfg.paths.inter_file == "inter.csv":
        alternative = raw_dir / "inter_final.csv"
        if alternative.exists():
            LOGGER.info("未找到 inter.csv，自动使用 inter_final.csv")
            inter_path = alternative
    user_path = raw_dir / cfg.paths.user_file

    books = standardize_frame(_read_csv(book_path), BOOK_ALIASES, ["book_id"], "book.csv")
    interactions = standardize_frame(
        _read_csv(inter_path), INTER_ALIASES, ["user_id", "book_id", "borrow_time"], inter_path.name
    )
    users = standardize_frame(_read_csv(user_path), USER_ALIASES, ["user_id"], "user.csv")

    for frame, id_columns in ((books, ["book_id"]), (users, ["user_id"]), (interactions, ["user_id", "book_id", "inter_id"])):
        for column in id_columns:
            frame[column] = _clean_string(frame[column])

    for column in ("title", "author", "publisher", "category1", "category2"):
        books[column] = _clean_string(books[column]).replace("", "未知")
    for column in ("gender", "dept", "grade", "user_type"):
        users[column] = _clean_string(users[column]).replace("", "未知")

    interactions["borrow_time"] = _parse_datetime(interactions["borrow_time"])
    interactions["return_time"] = _parse_datetime(interactions["return_time"])
    interactions["renew_time"] = _parse_datetime(interactions["renew_time"])
    interactions["renew_count"] = pd.to_numeric(interactions["renew_count"], errors="coerce").fillna(0).clip(lower=0)
    if (interactions["inter_id"] == "").all():
        interactions["inter_id"] = np.arange(len(interactions)).astype(str)

    before_clean = len(interactions)
    interactions = interactions.dropna(subset=["borrow_time"])
    interactions = interactions[(interactions["user_id"] != "") & (interactions["book_id"] != "")]
    interactions = interactions.drop_duplicates(subset=["user_id", "book_id", "borrow_time", "renew_count"])
    LOGGER.info("删除无效时间/重复记录：%d -> %d", before_clean, len(interactions))

    # 仅保留存在于交互中的实体；缺失元数据使用占位记录补齐。
    missing_books = sorted(set(interactions["book_id"]) - set(books["book_id"]))
    if missing_books:
        books = pd.concat([
            books,
            pd.DataFrame({
                "book_id": missing_books,
                "title": "未知", "author": "未知", "publisher": "未知",
                "category1": "未知", "category2": "未知",
            }),
        ], ignore_index=True)
    missing_users = sorted(set(interactions["user_id"]) - set(users["user_id"]))
    if missing_users:
        users = pd.concat([
            users,
            pd.DataFrame({
                "user_id": missing_users,
                "gender": "未知", "dept": "未知", "grade": "未知", "user_type": "未知",
            }),
        ], ignore_index=True)

    interactions = _iterative_kcore(
        interactions,
        int(cfg.preprocess.min_user_interactions),
        int(cfg.preprocess.min_item_interactions),
        int(cfg.preprocess.kcore_max_rounds),
    )
    active_users = sorted(interactions["user_id"].unique().tolist())
    active_books = sorted(interactions["book_id"].unique().tolist())
    users = users.drop_duplicates("user_id").set_index("user_id").reindex(active_users).fillna("未知").reset_index()
    books = books.drop_duplicates("book_id").set_index("book_id").reindex(active_books).fillna("未知").reset_index()

    user_to_idx = {value: idx for idx, value in enumerate(active_users)}
    item_to_idx = {value: idx for idx, value in enumerate(active_books)}
    interactions["user_idx"] = interactions["user_id"].map(user_to_idx).astype(np.int64)
    interactions["item_idx"] = interactions["book_id"].map(item_to_idx).astype(np.int64)
    users["user_idx"] = users["user_id"].map(user_to_idx).astype(np.int64)
    books["item_idx"] = books["book_id"].map(item_to_idx).astype(np.int64)

    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in interactions.groupby("user_idx", sort=False):
        train, valid, test = _split_user_history(group, str(cfg.preprocess.split_strategy))
        train_parts.append(train)
        valid_parts.append(valid)
        test_parts.append(test)
    train = pd.concat(train_parts, ignore_index=True) if train_parts else interactions.iloc[0:0]
    valid = pd.concat(valid_parts, ignore_index=True) if valid_parts else interactions.iloc[0:0]
    test = pd.concat(test_parts, ignore_index=True) if test_parts else interactions.iloc[0:0]

    reference_time = train["borrow_time"].max()
    for frame in (train, valid, test):
        if frame.empty:
            frame["weight"] = np.array([], dtype=np.float32)
            continue
        months = ((reference_time - frame["borrow_time"]).dt.days.clip(lower=0) / 30.0).astype(float)
        recency = np.power(float(cfg.preprocess.monthly_decay), months)
        renewal = 1.0 + np.log1p(frame["renew_count"].astype(float))
        frame["weight"] = (recency * renewal).astype(np.float32)

    user_features, user_transformers = _build_user_features(
        users, int(cfg.preprocess.metadata.user_svd_dim), int(cfg.project.seed)
    )
    item_features, item_transformers = _build_item_features(
        books,
        int(cfg.preprocess.metadata.item_svd_dim),
        int(cfg.preprocess.metadata.max_text_features),
        int(cfg.project.seed),
    )

    rows = train["user_idx"].to_numpy(np.int64)
    cols = train["item_idx"].to_numpy(np.int64)
    values = train["weight"].to_numpy(np.float32)
    train_matrix = sparse.csr_matrix(
        (values, (rows, cols)), shape=(len(users), len(books)), dtype=np.float32
    )
    binary_matrix = train_matrix.copy()
    binary_matrix.data[:] = 1.0

    popularity = np.asarray(binary_matrix.sum(axis=0)).ravel().astype(np.float32)
    renew_item = train.groupby("item_idx")["renew_count"].mean().reindex(range(len(books)), fill_value=0).to_numpy(np.float32)
    renew_user = train.groupby("user_idx")["renew_count"].mean().reindex(range(len(users)), fill_value=0).to_numpy(np.float32)

    _save_split(train, processed_dir / "train.csv")
    _save_split(valid, processed_dir / "valid.csv")
    _save_split(test, processed_dir / "test.csv")
    users.to_csv(processed_dir / "users.csv", index=False, encoding="utf-8-sig")
    books.to_csv(processed_dir / "books.csv", index=False, encoding="utf-8-sig")
    sparse.save_npz(processed_dir / "train_matrix.npz", train_matrix)
    sparse.save_npz(processed_dir / "train_binary_matrix.npz", binary_matrix)
    np.save(processed_dir / "user_features.npy", user_features)
    np.save(processed_dir / "item_features.npy", item_features)
    np.save(processed_dir / "popularity.npy", popularity)
    np.save(processed_dir / "renew_item.npy", renew_item)
    np.save(processed_dir / "renew_user.npy", renew_user)
    save_pickle(user_transformers, processed_dir / "user_feature_transformers.pkl")
    save_pickle(item_transformers, processed_dir / "item_feature_transformers.pkl")
    save_json(
        {
            "user_to_idx": user_to_idx,
            "item_to_idx": item_to_idx,
            "idx_to_user": active_users,
            "idx_to_item": active_books,
        },
        processed_dir / "mappings.json",
    )
    save_json(
        {
            "num_users": len(users),
            "num_items": len(books),
            "num_interactions": len(interactions),
            "num_train": len(train),
            "num_valid": len(valid),
            "num_test": len(test),
            "user_feature_dim": int(user_features.shape[1]),
            "item_feature_dim": int(item_features.shape[1]),
            "split_strategy": str(cfg.preprocess.split_strategy),
            "reference_time": str(reference_time),
        },
        processed_dir / "meta.json",
    )

    LOGGER.info(
        "预处理完成：users=%d, items=%d, train=%d, valid=%d, test=%d",
        len(users), len(books), len(train), len(valid), len(test),
    )
