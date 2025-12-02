# modern_methods.py
# Part B: 現代 AI 方法（GPT-4o & Embedding）
#
# 功能：
#   B-1: 語意相似度計算 (embedding + cosine similarity)
#   B-2: AI 文本分類（情緒 + 主題 + 信心分數）
#   B-3: AI 自動摘要（可控制長度）
#
# 使用方式：
#   1. 先在系統環境變數設定 OPENAI_API_KEY
#   2. 在終端機執行：python modern_methods.py
#
# 注意：這支檔案只負責「呼叫 OpenAI API」，量化比較放在 comparison.py 處理

from __future__ import annotations

import math
import os
import json
from typing import List, Dict, Any, Optional

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity


# ===== 共用：Client & Cache =====

# 簡單快取，避免重複呼叫浪費 token
_EMBEDDING_CACHE: Dict[str, List[float]] = {}
_CLASSIFY_CACHE: Dict[str, Dict[str, Any]] = {}
_SUMMARY_CACHE: Dict[str, str] = {}


def get_client(api_key: Optional[str] = None) -> OpenAI:
    """
    建立 OpenAI Client
    如果 api_key 沒給，會從環境變數 OPENAI_API_KEY 讀取
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "找不到 API key，請在程式參數帶入 api_key，或在環境變數設定 OPENAI_API_KEY"
        )
    return OpenAI(api_key=key)


# ===== B-1: 語意相似度計算（Embedding） =====

def get_embedding(client: OpenAI, text: str,
                  model: str = "text-embedding-3-small") -> List[float]:
    """
    取得單一句子的 embedding 向量
    預設使用 text-embedding-3-small，成本低、效能夠用。:contentReference[oaicite:1]{index=1}
    """
    if text in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[text]

    resp = client.embeddings.create(
        model=model,
        input=text,
    )
    # 新版 API 回傳 data[0].embedding
    vec = resp.data[0].embedding
    _EMBEDDING_CACHE[text] = vec
    return vec


def cosine_sim(vec1: List[float], vec2: List[float]) -> float:
    """
    手刻 cosine similarity，避免額外依賴
    """
    if len(vec1) != len(vec2):
        raise ValueError("向量維度不一致")

    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def gpt_similarity(text1: str, text2: str, api_key: Optional[str] = None) -> float:
    """
    B-1: 使用 GPT Embedding 計算語意相似度
    回傳值介於 -1 ~ 1，通常在 0~1 之間。
    """
    client = get_client(api_key)

    emb1 = get_embedding(client, text1)
    emb2 = get_embedding(client, text2)

    sim = cosine_sim(emb1, emb2)
    return sim


# ===== B-2: AI 文本分類（情緒 + 主題 + 信心分數） =====

CLASSIFICATION_SYSTEM_PROMPT = """你是一個中文文本分類器，請依照規則輸出 JSON。

任務說明：
1. sentiment（情緒）：從 ["正面", "負面", "中性"] 中選一個
2. topic（主題類別）：從 ["科技", "運動", "美食", "旅遊", "其他"] 中選一個
3. confidence：0 到 1 之間的小數（例如 0.85）

請「只輸出 JSON」，格式例如：
{"sentiment": "正面", "topic": "美食", "confidence": 0.92}
"""


def ai_classify(text: str, api_key: Optional[str] = None,
                model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    B-2: 使用 GPT-4o / 4o-mini 進行多維度分類:contentReference[oaicite:2]{index=2}

    回傳格式：
    {
        "sentiment": "正面/負面/中性",
        "topic": "科技/運動/美食/旅遊/其他",
        "confidence": 0.0 ~ 1.0
    }
    """
    if text in _CLASSIFY_CACHE:
        return _CLASSIFY_CACHE[text]

    client = get_client(api_key)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.0,  # 分類任務不需要創造性
    )

    content = resp.choices[0].message.content.strip()

    # 安全起見：若模型沒完全照 JSON 格式，我們做一點修正
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # 粗略修補：找出第一對大括號
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            json_str = content[start : end + 1]
            result = json.loads(json_str)
        else:
            # 實在救不回來就給預設值
            result = {
                "sentiment": "中性",
                "topic": "其他",
                "confidence": 0.0,
                "raw": content,
            }

    # 補上缺欄位 & 做基本檢查
    sentiment = result.get("sentiment", "中性")
    topic = result.get("topic", "其他")
    confidence = float(result.get("confidence", 0.0))

    result_clean = {
        "sentiment": sentiment,
        "topic": topic,
        "confidence": confidence,
    }

    _CLASSIFY_CACHE[text] = result_clean
    return result_clean


# ===== B-3: AI 自動摘要 =====

SUMMARY_SYSTEM_PROMPT = """你是一個專門為中文文章做重點整理的助手。

任務要求：
1. 生成「摘要」，而不是逐句翻譯或逐句重述。
2. 盡量保留關鍵資訊與因果關係。
3. 摘要語氣維持中立、客觀、書面中文。
4. 不要加入原文沒有的新資訊。
"""


def ai_summarize(text: str,
                 max_length: int = 120,
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini") -> str:
    """
    B-3: 使用 GPT-4o 產生中文摘要:contentReference[oaicite:3]{index=3}

    參數：
      - text: 原文
      - max_length: 希望摘要上限字數（粗略控制）
    """
    cache_key = f"{hash(text)}-{max_length}"
    if cache_key in _SUMMARY_CACHE:
        return _SUMMARY_CACHE[cache_key]

    client = get_client(api_key)

    user_prompt = f"請為以下文章產生精簡摘要，盡量控制在約 {max_length} 個中文字以內：\n\n{text}"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    summary = resp.choices[0].message.content.strip()
    # 簡單做一次長度控制（太長就截斷）
    if len(summary) > max_length * 1.3:
        summary = summary[: max_length]

    _SUMMARY_CACHE[cache_key] = summary
    return summary


# ===== 測試用 main =====

def _demo():
    """
    簡單在終端機測試 B-1 / B-2 / B-3
    """
    print("=== 測試 B-1：語意相似度 ===")
    t1 = "人工智慧正在改變世界，許多工作流程開始被自動化。"
    t2 = "AI 技術讓許多重複性工作可以交給電腦執行。"
    t3 = "今天的天氣很好，適合出去運動。"

    sim_12 = gpt_similarity(t1, t2)
    sim_13 = gpt_similarity(t1, t3)
    print(f"t1 vs t2 相似度：{sim_12:.4f}")
    print(f"t1 vs t3 相似度：{sim_13:.4f}")

    print("\n=== 測試 B-2：AI 文本分類 ===")
    texts = [
        "這家餐廳的牛肉麵超好吃，服務也很貼心，下次還會再來。",
        "最近 AI 技術發展很快，很多公司都在談數據與演算法。",
        "每天運動雖然有點累，但體力真的有變好。",
        "這次出國旅遊的飯店很乾淨，景點也安排得不錯。",
    ]
    for i, txt in enumerate(texts, 1):
        res = ai_classify(txt)
        print(f"[句子{i}] {txt}")
        print("  ->", res)

    print("\n=== 測試 B-3：AI 自動摘要 ===")
    article = (
        "人工智慧（AI）的快速發展，正在改變我們的生活與工作方式。"
        "從智慧客服、語音助理，到自動駕駛與醫療診斷，"
        "越來越多領域開始導入 AI 技術，以提升效率與準確度。"
        "然而，AI 的廣泛應用也帶來隱私、偏見與責任歸屬等議題，"
        "如何在創新與風險之間取得平衡，成為各界關注的重點。"
    )
    summary = ai_summarize(article, max_length=80)
    print("【原文】", article)
    print("【AI 摘要】", summary)


if __name__ == "__main__":
    _demo()

