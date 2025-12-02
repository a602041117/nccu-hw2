# comparison.py
# Part C: 傳統方法 vs 現代 AI 方法的比較實驗
#
# 輸出：
#   results/performance_metrics.json
#   results/classification_results.csv
#   results/summarization_comparison.txt

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from traditional_methods import (
    RuleBasedSentimentClassifier,
    TopicClassifier,
    StatisticalSummarizer,
)
from modern_methods import (
    gpt_similarity,
    ai_classify,
    ai_summarize,
)

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)


# ====== 一些小工具 ======

@dataclass
class MethodResult:
    accuracy: float | None = None
    time_sec: float | None = None
    extra: Dict[str, Any] | None = None


def tfidf_pair_similarity(text1: str, text2: str) -> float:
    """
    對兩句文本用 TF-IDF (char level) 計算餘弦相似度
    """
    vectorizer = TfidfVectorizer(analyzer="char")
    tfidf = vectorizer.fit_transform([text1, text2])
    sim_matrix = cosine_similarity(tfidf)
    return float(sim_matrix[0, 1])


# ====== 1. 相似度任務：TF-IDF vs GPT Embedding ======

def evaluate_similarity(api_key: str | None = None) -> Dict[str, Any]:
    """
    相似度任務：
      - 準備一組 (text1, text2, label)：
        label = 1 表示語意相似、0 表示不相似
      - 以門檻將連續相似度 -> 二元分類，計算準確率
    """

    # 你可以依自己需求修改這些測試句子與標註
    pairs = [
        (
            "人工智慧正在改變工作流程，許多重複性工作被自動化。",
            "AI 技術讓企業能自動處理繁瑣的行政工作。",
            1,
        ),
        (
            "今天的天氣很好，適合出去運動。",
            "我打算晚上去跑步和做重量訓練。",
            1,
        ),
        (
            "這家餐廳的牛肉麵很好吃，下次還想再來。",
            "這次的會議討論了雲端架構的設計與維運。",
            0,
        ),
        (
            "機器學習模型需要大量數據來訓練。",
            "我最近開始學做甜點，覺得烘焙很療癒。",
            0,
        ),
    ]

    # ===== TF-IDF 版本 =====
    tfidf_threshold = 0.3  # 你可以之後調整門檻

    t0 = time.perf_counter()
    tfidf_scores = [tfidf_pair_similarity(t1, t2) for (t1, t2, _) in pairs]
    t1 = time.perf_counter()

    tfidf_preds = [1 if s >= tfidf_threshold else 0 for s in tfidf_scores]
    labels = [lab for _, _, lab in pairs]
    tfidf_correct = sum(int(p == y) for p, y in zip(tfidf_preds, labels))
    tfidf_acc = tfidf_correct / len(labels)

    tfidf_result = MethodResult(
        accuracy=tfidf_acc,
        time_sec=t1 - t0,
        extra={
            "threshold": tfidf_threshold,
            "scores": tfidf_scores,
        },
    )

    # ===== GPT Embedding 版本 =====
    gpt_threshold = 0.7  # GPT 的語意相似度通常比 TF-IDF 高

    t0 = time.perf_counter()
    gpt_scores = [gpt_similarity(t1, t2, api_key=api_key) for (t1, t2, _) in pairs]
    t1 = time.perf_counter()

    gpt_preds = [1 if s >= gpt_threshold else 0 for s in gpt_scores]
    gpt_correct = sum(int(p == y) for p, y in zip(gpt_preds, labels))
    gpt_acc = gpt_correct / len(labels)

    gpt_result = MethodResult(
        accuracy=gpt_acc,
        time_sec=t1 - t0,
        extra={
            "threshold": gpt_threshold,
            "scores": gpt_scores,
        },
    )

    return {
        "pairs": [
            {
                "text1": t1,
                "text2": t2,
                "label": lab,
                "tfidf_score": s_tfidf,
                "gpt_score": s_gpt,
            }
            for (t1, t2, lab), s_tfidf, s_gpt in zip(
                pairs, tfidf_scores, gpt_scores
            )
        ],
        "tfidf": asdict(tfidf_result),
        "gpt": asdict(gpt_result),
    }


# ====== 2. 文本分類任務：規則式 vs GPT ======

def evaluate_classification(api_key: str | None = None) -> Dict[str, Any]:
    """
    文本分類任務：
      - 情緒 sentiment: 正面/負面/中性
      - 主題 topic: 科技/運動/美食/旅遊/其他
    """

    # 測試資料 + 人工標記（你可以依作業調整）
    data = [
        {
            "text": "這家餐廳的牛肉麵超好吃，服務也很貼心，下次還會再來。",
            "sentiment": "正面",
            "topic": "美食",
        },
        {
            "text": "最近 AI 技術發展很快，很多公司都在談數據與演算法。",
            "sentiment": "正面",
            "topic": "科技",
        },
        {
            "text": "今天開會一直改需求，進度被拖延，心情有點差。",
            "sentiment": "負面",
            "topic": "科技",
        },
        {
            "text": "每天運動雖然有點累，但體力真的有變好。",
            "sentiment": "正面",
            "topic": "運動",
        },
        {
            "text": "這次出國旅遊的飯店很乾燥，服務也不太友善，有點失望。",
            "sentiment": "負面",
            "topic": "旅遊",
        },
    ]

    rule_sentiment_clf = RuleBasedSentimentClassifier()
    rule_topic_clf = TopicClassifier()

    # ==== 規則式 ====
    t0 = time.perf_counter()
    for item in data:
        item["rule_sentiment"] = rule_sentiment_clf.classify(item["text"])
        item["rule_topic"] = rule_topic_clf.classify(item["text"])
    t1 = time.perf_counter()
    rule_time = t1 - t0

    # ==== GPT 分類 ====
    t0 = time.perf_counter()
    for item in data:
        res = ai_classify(item["text"], api_key=api_key)
        item["gpt_sentiment"] = res["sentiment"]
        item["gpt_topic"] = res["topic"]
        item["gpt_confidence"] = res["confidence"]
    t1 = time.perf_counter()
    gpt_time = t1 - t0

    # ==== 準確率計算 ====
    def acc(pred_key: str, label_key: str) -> float:
        correct = sum(1 for item in data if item[pred_key] == item[label_key])
        return correct / len(data)

    rule_sent_acc = acc("rule_sentiment", "sentiment")
    rule_topic_acc = acc("rule_topic", "topic")
    gpt_sent_acc = acc("gpt_sentiment", "sentiment")
    gpt_topic_acc = acc("gpt_topic", "topic")

    # 輸出 CSV
    csv_path = RESULT_DIR / "classification_results.csv"
    import csv

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "text",
                "label_sentiment",
                "label_topic",
                "rule_sentiment",
                "rule_topic",
                "gpt_sentiment",
                "gpt_topic",
                "gpt_confidence",
            ]
        )
        for item in data:
            writer.writerow(
                [
                    item["text"],
                    item["sentiment"],
                    item["topic"],
                    item["rule_sentiment"],
                    item["rule_topic"],
                    item["gpt_sentiment"],
                    item["gpt_topic"],
                    f"{item['gpt_confidence']:.3f}",
                ]
            )

    return {
        "rule_based": {
            "accuracy_sentiment": rule_sent_acc,
            "accuracy_topic": rule_topic_acc,
            "time_sec": rule_time,
        },
        "gpt": {
            "accuracy_sentiment": gpt_sent_acc,
            "accuracy_topic": gpt_topic_acc,
            "time_sec": gpt_time,
        },
    }


# ====== 3. 摘要任務：統計式 vs GPT ======

def evaluate_summarization(api_key: str | None = None) -> Dict[str, Any]:
    """
    摘要任務：
      - 比較傳統統計式摘要（StatisticalSummarizer） vs GPT 摘要
      - 量化部分需人工評分，因此這裡輸出對照檔方便你打分
    """

    articles = [
        """
        人工智慧（AI）的發展正在改變我們的生活方式，
        從早上起床時的智慧鬧鐘，到通勤時的路線規劃，
        再到工作中的各種輔助工具，AI 無處不在。
        在醫療領域，AI 協助醫生進行疾病診斷，提高了診斷的準確率和效率。
        然而，AI 的快速發展也帶來了一些挑戰。
        其中一個重要議題是隱私和安全，AI 系統需要大量數據來訓練，
        如何保護個人資料成為關鍵問題。
        最後是倫理與責任歸屬，AI 的決策過程往往缺乏透明度，
        可能會產生偏見或歧視。
        只有在兼顧創新與風險的前提下，AI 技術才能真正為人類福祉服務。
        """,
        """
        規律運動對身體健康有許多好處。
        研究指出，每週進行三到五次中等強度的運動，
        可以有效降低心血管疾病、糖尿病與肥胖的風險。
        除了生理上的益處，運動也能改善心理健康，
        例如減少壓力、焦慮與憂鬱情緒，並提升睡眠品質。
        然而，許多人因為工作忙碌或缺乏動機而難以維持運動習慣。
        因此，找到自己喜歡且容易持之以恆的運動方式，
        比追求高強度、短期的訓練更為重要。
        """,
    ]

    stat_summarizer = StatisticalSummarizer()

    lines: List[str] = []
    t0 = time.perf_counter()
    stat_summaries = [stat_summarizer.summarize(a, ratio=0.3) for a in articles]
    t1 = time.perf_counter()
    stat_time = t1 - t0

    t0 = time.perf_counter()
    gpt_summaries = [ai_summarize(a, max_length=120, api_key=api_key) for a in articles]
    t1 = time.perf_counter()
    gpt_time = t1 - t0

    # 輸出對照檔，方便你人工打分（資訊保留度 / 語句通順度）
    txt_path = RESULT_DIR / "summarization_comparison.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        for idx, (orig, stat_sum, gpt_sum) in enumerate(
            zip(articles, stat_summaries, gpt_summaries), start=1
        ):
            f.write(f"=== 文章 {idx} ===\n")
            f.write("[原文]\n")
            f.write(orig.strip() + "\n\n")
            f.write("[統計式摘要]\n")
            f.write(stat_sum.replace("\n", "") + "\n\n")
            f.write("[GPT 摘要]\n")
            f.write(gpt_sum.replace("\n", "") + "\n\n")
            f.write("[請自行評分] 資訊保留度(0-100%) / 語句通順度(1-5)：\n\n\n")

    return {
        "statistical": {
            "time_sec": stat_time,
        },
        "gpt": {
            "time_sec": gpt_time,
        },
        "note": "資訊保留度與語句通順度請參考 summarization_comparison.txt 進行人工評分。",
    }


# ====== 主程式 ======

def main(api_key: str | None = None):
    metrics: Dict[str, Any] = {}

    print(">>> 評估 相似度 任務...")
    metrics["similarity"] = evaluate_similarity(api_key=api_key)

    print(">>> 評估 文本分類 任務...")
    metrics["classification"] = evaluate_classification(api_key=api_key)

    print(">>> 評估 摘要 任務...")
    metrics["summarization"] = evaluate_summarization(api_key=api_key)

    out_path = RESULT_DIR / "performance_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n所有結果已輸出到 {out_path}")
    print("你也可以查看 classification_results.csv 和 summarization_comparison.txt")


if __name__ == "__main__":
    # 若你已經在環境變數設定 OPENAI_API_KEY，就可以不傳 api_key
    main(api_key=None)
