from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


BOOK_ALIASES = {
    "book_id": ["book_id", "BOOK_ID", "图书ID", "图书id", "书目ID", "书目编号"],
    "title": ["题名", "书名", "title", "book_title"],
    "author": ["作者", "author"],
    "publisher": ["出版社", "publisher"],
    "category1": ["一级分类", "一级类别", "category1", "first_category"],
    "category2": ["二级分类", "二级类别", "category2", "second_category"],
}

INTER_ALIASES = {
    "inter_id": ["inter_id", "INTER_ID", "交互ID", "记录ID"],
    "user_id": ["user_id", "USER_ID", "借阅人", "用户ID", "读者ID", "读者证号"],
    "book_id": ["book_id", "BOOK_ID", "图书ID", "图书id", "书目ID", "书目编号"],
    "borrow_time": ["借阅时间", "借书时间", "borrow_time", "borrow_date"],
    "return_time": ["还书时间", "return_time", "return_date"],
    "renew_time": ["续借时间", "renew_time", "renew_date"],
    "renew_count": ["续借次数", "renew_count", "renew_num"],
}

USER_ALIASES = {
    "user_id": ["借阅人", "user_id", "USER_ID", "用户ID", "读者ID", "读者证号"],
    "gender": ["性别", "gender", "sex"],
    "dept": ["DEPT", "dept", "院系", "学院", "部门"],
    "grade": ["年级", "grade"],
    "user_type": ["类型", "user_type", "读者类型", "学历层次"],
}


@dataclass(frozen=True)
class ResolvedSchema:
    rename_map: dict[str, str]
    missing_required: list[str]


def _normalize_name(value: object) -> str:
    return str(value).strip().replace("\ufeff", "")


def resolve_schema(
    frame: pd.DataFrame,
    aliases: dict[str, list[str]],
    required: Iterable[str],
) -> ResolvedSchema:
    normalized_to_original = {_normalize_name(col).lower(): col for col in frame.columns}
    rename_map: dict[str, str] = {}
    missing: list[str] = []

    for standard_name, candidates in aliases.items():
        matched = None
        for candidate in candidates:
            original = normalized_to_original.get(_normalize_name(candidate).lower())
            if original is not None:
                matched = original
                break
        if matched is not None:
            rename_map[matched] = standard_name
        elif standard_name in required:
            missing.append(standard_name)
    return ResolvedSchema(rename_map=rename_map, missing_required=missing)


def standardize_frame(
    frame: pd.DataFrame,
    aliases: dict[str, list[str]],
    required: Iterable[str],
    table_name: str,
) -> pd.DataFrame:
    schema = resolve_schema(frame, aliases, required)
    if schema.missing_required:
        raise ValueError(
            f"{table_name} 缺少必要字段 {schema.missing_required}。"
            f"当前列为：{list(frame.columns)}"
        )
    result = frame.rename(columns=schema.rename_map).copy()
    for standard_name in aliases:
        if standard_name not in result.columns:
            result[standard_name] = ""
    return result[list(aliases.keys())]
