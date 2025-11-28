#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2025/10/25 14:26
# @Author : Hiking
# @File : app_v3.py


from fastapi import FastAPI
from pydantic import BaseModel, RootModel
from typing import List, Optional
import httpx
import numpy as np
import asyncio
import logging
import Levenshtein
from rapidfuzz import process, fuzz
import time
import re

logging.basicConfig(level=logging.INFO)

# 组织名称规范化配置
COMMON_ORG_SUFFIXES = [
    "股份有限公司",
    "有限责任公司",
    "集团有限责任公司",
    "集团有限公司",
    "有限公司",
    "股份公司",
    "有限合伙",
]
REMOVE_CHAR_PATTERN = re.compile(r"[()\uff08\uff09\s·•．.]+")

# ================= 基础配置 =================
app = FastAPI(title="通用纠错接口", version="v1.0.0")

LLM_API_URL = "http://10.60.44.83:8980/v1/chat/completions"
LLM_MODEL_NAME = "qwen3-next-80b-a3b-instruct"
LLM_API_KEY = "gpustack_00da2eb6c2965c65_240762a0ead8947852987f7b65b36654"
EMBED_MODEL_NAME = "nlp_gte_sentence-embedding_chinese-large"
EMBED_API_URL = "http://10.60.44.51:8980/v1/embeddings"
API_KEY = "gpustack_f2c2a286a8acbd80_2e1b425b25e7ff54ce94a0e2d9746a8b"
Timeout = 10

# 复用 httpx 异步客户端
app.state.http_client = httpx.AsyncClient(timeout=Timeout)


# ================= 数据模型 =================
class TransItem(BaseModel):
    key: str
    value: str


class CorrectionRequestItem(BaseModel):
    needTransKey: str
    needTransValue: str
    transList: List[TransItem]


class CorrectionResponseItem(BaseModel):
    needTransKey: str
    needTransValue: str
    transValue: TransItem


class ErrorCorrectionRequest(RootModel):
    root: List[CorrectionRequestItem]


class ErrorCorrectionResponse(BaseModel):
    error_no: int
    error_info: str
    res: List[CorrectionResponseItem]


# ================== 工具函数 ==================


def normalize_org_name(name: Optional[str]) -> str:
    """去除括号、空白以及尾部的通用公司后缀，提取组织名称核心部分"""
    if not name:
        return ""
    normalized = REMOVE_CHAR_PATTERN.sub("", str(name).strip())
    if not normalized:
        return ""

    # 迭代剥离常见的公司后缀（例如“股份有限公司”、“有限责任公司”等）
    # 只要仍有匹配就持续移除，可以覆盖“集团有限公司”等复合后缀
    while True:
        stripped = False
        for suffix in COMMON_ORG_SUFFIXES:
            if normalized.endswith(suffix) and len(normalized) > len(suffix):
                normalized = normalized[: -len(suffix)]
                stripped = True
                break
        if not stripped:
            break
    return normalized


def build_match_key(text: Optional[str]) -> str:
    """生成用于匹配的key，优先返回规范化结果，若为空则退回简单清洗后的字符串"""
    normalized = normalize_org_name(text)
    if normalized:
        return normalized
    if not text:
        return ""
    fallback = REMOVE_CHAR_PATTERN.sub("", str(text).strip())
    return fallback


def edit_distance_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    dist = Levenshtein.distance(a, b)
    max_len = max(len(a), len(b))
    return 1 - dist / max_len


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


async def get_embeddings(texts: List[str]) -> np.ndarray:
    """调用嵌入模型接口获取文本向量（全局client复用）"""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": EMBED_MODEL_NAME, "input": texts}
    client: httpx.AsyncClient = app.state.http_client
    response = await client.post(EMBED_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    embeddings = [item["embedding"] for item in data["data"]]
    return np.array(embeddings)


# ========== 调用大模型判断最匹配项 ==========
async def choose_best_by_llm(need: str, candidates: List[str]) -> str:
    """
    通过大模型判断最符合语义的候选项
    """
    # 限制候选数量，防止提示过长
    # candidates = candidates[:20]

    candidate_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])

    prompt = f"""
        你是一个智能匹配助手。请在候选项中选择最符合输入语义的一项。
        输入: "{need}"
        候选项:
        {candidate_text}
        请仅返回最符合语义的一项的原始文本，不要返回解释或编号。
        """
    # logging.info(prompt)
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }

    async with httpx.AsyncClient(timeout=Timeout) as client:
        try:
            response = await client.post(LLM_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            result = data["choices"][0]["message"]["content"].strip()
            return result
        except Exception as e:
            logging.warning(f"LLM匹配调用失败: {e}")
            return candidates[0]  # fallback


# ================== 核心逻辑 ==================
async def correct_one_string(
    need: str,
    trans_items: List[TransItem],
    top_k: int = 200,
    top_p: float = 0.2,
    mode="llm",
) -> TransItem:
    """
    基于多级匹配策略为输入字符串找到最合适的翻译项

    该函数采用分层匹配策略：
    1. 完全匹配（最快）case1
    2. 编辑距离匹配（初筛） case2；后期可考虑tf-idf加权的编辑距离，关注重点词汇
    3. 语义匹配（准确匹配） case2 其中实现了两种mode mode=llm基于大模型+prompt； mode=embedding 基于嵌入模型+cos_sim,可自行切换
    4. 接口延时：在16W数据集下测试，单次请求耗时0.5640s =>   INFO:root:✅ 建行 match 单次匹配耗时: 0.5640s


    Args:
        need (str): 需要匹配的输入字符串
        trans_items (List[TransItem]): 候选翻译项列表
        top_k (int, optional): 编辑距离筛选的候选数量上限，默认200
        top_p (float, optional): 编辑距离得分阈值比例(0-1)，用于筛选最低相似度，默认0.2
        mode (str, optional): 语义匹配模式，可选"llm"或"embedding"，默认"embedding"

    Returns:
        TransItem: 需要匹配的翻译项

    Raises:
        不会显式抛出异常，但会在匹配失败时使用fallback机制返回默认项
    """

    t0 = time.perf_counter()
    # 1️⃣ 构建索引（保留原始TransItem用于返回，仅对匹配key做规范化）
    trans_item_map: dict[str, TransItem] = {}
    for item in trans_items:
        match_key = build_match_key(item.key)
        if not match_key:
            continue
        trans_item_map.setdefault(match_key, item)
    patterns = list(trans_item_map.keys())

    if not patterns:
        logging.warning(f"待检测字符串:{need}=>候选列表为空，直接返回默认值")
        return trans_items[0] if trans_items else TransItem(key="", value="")

    need_key = build_match_key(need)
    query = need_key or (need.strip() if need else "")

    # 2️⃣ 完全匹配快速返回
    if need_key and need_key in trans_item_map:
        logging.info(f"待检测字符串:{need}=>符合完全匹配逻辑")
        return trans_item_map[need_key]

    if not query:
        logging.info(f"待检测字符串:{need}=>规范化后为空，返回默认值")
        return trans_items[0] if trans_items else TransItem(key="", value="")

    logging.info(f"待检测字符串:{need}=>不符合完全匹配逻辑")
    # 3️⃣ 编辑距离筛选 top-k 候选
    filter_patterns = process.extract(
        query,
        patterns,
        scorer=fuzz.ratio if not re.search(r"银行", need or "") else fuzz.partial_ratio,
        limit=top_k,
        score_cutoff=int(top_p * 100),
    )
    candidates_patterns = [x[0] for x in filter_patterns]

    if not candidates_patterns:
        # fallback
        logging.info(f"待检测字符串:{need}=>没有找到编辑距离候选集")
        return trans_items[0]

    # 4️⃣ 获取候选项（O(1) 查找）
    candidates_items = [
        trans_item_map[p] for p in candidates_patterns if p in trans_item_map
    ]

    # 5️⃣ 语义匹配
    try:
        match mode:
            case "llm":
                logging.info(f"{need} match with # ==== 基于LLM的语义判断 ====")
                best_match = await choose_best_by_llm(
                    need, [item.key for item in candidates_items]
                )
                logging.info(f"{need} matched {best_match}")
                # 找到最匹配的TransItem
                for item in candidates_items:
                    if best_match.strip() == item.key.strip():
                        return item
                # 若模型输出不完全匹配，尝试模糊查找
                matched = process.extractOne(
                    best_match, [i.key for i in candidates_items], scorer=fuzz.ratio
                )
                if matched:
                    idx = [i.key for i in candidates_items].index(matched[0])
                    return candidates_items[idx]
                return candidates_items[0]
            case "embedding":
                logging.info(f"{need} match with  # == == 基于嵌入模型语义判断 == ==")
                texts = [need] + candidates_patterns
                embeds = await get_embeddings(texts)
                query_vec, cand_vecs = embeds[0], embeds[1:]
                sims = np.dot(cand_vecs, query_vec) / (
                    np.linalg.norm(cand_vecs, axis=1) * np.linalg.norm(query_vec)
                )
                best_idx = int(np.argmax(sims))
                return candidates_items[best_idx]
            case _:
                return candidates_items[0]
    except Exception as e:
        logging.warning(f"llm/embedding语义匹配失败，fallback到编辑距离: {e}")
        return candidates_items[0] if candidates_items else trans_items[0]
    finally:
        t1 = time.perf_counter()
        logging.info(f"✅ {need} match 单次匹配耗时: {t1 - t0:.4f}s")


# ================== 接口实现 ==================


@app.post("/v1/error_correction", response_model=ErrorCorrectionResponse)
async def error_correction(request: ErrorCorrectionRequest):
    t0 = time.perf_counter()
    try:
        request_items = request.root

        # 并发执行全部请求
        tasks = [
            correct_one_string(item.needTransValue, item.transList)
            for item in request_items
        ]
        trans_values = await asyncio.gather(*tasks)

        response_items = [
            CorrectionResponseItem(
                needTransKey=req_item.needTransKey,
                needTransValue=req_item.needTransValue,
                transValue=trans_value,
            )
            for req_item, trans_value in zip(request_items, trans_values)
        ]

        t1 = time.perf_counter()
        logging.info(f"✨ 整体匹配耗时: {t1 - t0:.3f}s")

        return ErrorCorrectionResponse(
            error_no=0,
            error_info="success",
            res=response_items,
        )

    except Exception as e:
        logging.exception("服务异常")
        response_items = [
            CorrectionResponseItem(
                needTransKey=item.needTransKey,
                needTransValue=item.needTransValue,
                transValue=(
                    item.transList[0] if item.transList else TransItem(key="", value="")
                ),
            )
            for item in request.root
        ]
        return ErrorCorrectionResponse(
            error_no=500,
            error_info=str(e),
            res=response_items,
        )


# ================== 启动入口 ==================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8085)
