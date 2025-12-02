# traditional_methods.py
# A-1: TF-IDF + A-2: 規則式分類

import math
from collections import Counter
from typing import Dict, List

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re
# 上面檔案一開始應該已經有 import re，如果沒有記得加

# ---------- A-1: 手刻 TF / IDF / TF-IDF ----------

def calculate_tf(word_dict: Dict[str, int], total_words: int) -> Dict[str, float]:
    tf_dict: Dict[str, float] = {}
    for word, count in word_dict.items():
        tf_dict[word] = count / total_words if total_words > 0 else 0.0
    return tf_dict


def calculate_idf(documents_tokens: List[List[str]]) -> Dict[str, float]:
    num_docs = len(documents_tokens)
    doc_freq: Dict[str, int] = {}

    for doc in documents_tokens:
        unique_words = set(doc)
        for w in unique_words:
            doc_freq[w] = doc_freq.get(w, 0) + 1

    idf: Dict[str, float] = {}
    for word, df in doc_freq.items():
        idf[word] = math.log(num_docs / (df + 1))

    return idf


def calculate_tfidf(documents: List[str]) -> List[Dict[str, float]]:
    tokenized_docs: List[List[str]] = []
    for doc in documents:
        tokens = list(jieba.cut(doc))
        tokenized_docs.append(tokens)

    idf = calculate_idf(tokenized_docs)

    tfidf_results: List[Dict[str, float]] = []
    for tokens in tokenized_docs:
        word_dict = Counter(tokens)
        total_words = len(tokens)
        tf = calculate_tf(word_dict, total_words)

        tfidf: Dict[str, float] = {}
        for word in tf:
            tfidf[word] = tf[word] * idf.get(word, 0.0)
        tfidf_results.append(tfidf)

    return tfidf_results


def cosine_sim_from_dicts(tfidf_list: List[Dict[str, float]]) -> np.ndarray:
    vocab = sorted({w for d in tfidf_list for w in d.keys()})
    matrix = []
    for d in tfidf_list:
        vec = [d.get(w, 0.0) for w in vocab]
        matrix.append(vec)
    matrix = np.array(matrix)
    sim_matrix = cosine_similarity(matrix)
    return sim_matrix


def sklearn_tfidf_similarity(documents: List[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(analyzer="char")
    tfidf_matrix = vectorizer.fit_transform(documents)
    sim_matrix = cosine_similarity(tfidf_matrix)
    return sim_matrix


# ---------- A-2: 規則式情緒分類 ----------

class RuleBasedSentimentClassifier:
    def __init__(self):
        self.positive_words = ["好", "棒", "優秀", "喜歡", "推薦", "滿意", "開心", "值得", "精彩", "完美"]
        self.negative_words = ["差", "糟", "失望", "討厭", "不推薦", "浪費", "無聊", "爛", "糟糕", "差勁"]
        self.negation_words = ["不", "沒", "無", "非", "別"]

    def classify(self, text: str) -> str:
        score = 0

        for w in self.positive_words:
            score += text.count(w)
        for w in self.negative_words:
            score -= text.count(w)

        for neg in self.negation_words:
            for w in self.positive_words:
                if neg + w in text:
                    score -= 2
            for w in self.negative_words:
                if neg + w in text:
                    score += 2

        if score > 0:
            return "正面"
        elif score < 0:
            return "負面"
        else:
            return "中性"


# ---------- A-2: 規則式主題分類 ----------

class TopicClassifier:
    def __init__(self):
        self.topic_keywords = {
            "科技": ["AI", "人工智慧", "電腦", "軟體", "程式", "演算法"],
            "運動": ["運動", "健身", "跑步", "比賽", "游泳", "球類"],
            "美食": ["吃", "食物", "餐廳", "美味", "料理", "烹飪"],
            "旅遊": ["旅遊", "景點", "飯店", "機票", "度假", "觀光"],
        }

    def classify(self, text: str) -> str:
        scores = {topic: 0 for topic in self.topic_keywords}

        for topic, keywords in self.topic_keywords.items():
            for kw in keywords:
                scores[topic] += text.count(kw)

        # 如果所有分數都是 0 → 回傳 "其他"
        if all(score == 0 for score in scores.values()):
            return "其他"

        # 有命中 → 回傳最高分的主題
        best_topic = max(scores, key=scores.get)
        return best_topic

# ---------- A-3: 統計式自動摘要 ----------

class StatisticalSummarizer:
    def __init__(self):
        # 停用詞可以再依需求擴充
        self.stop_words = set(["的", "了", "在", "是", "和", "也", "就", "都", "而且", "但是", "如果"])

    def _tokenize(self, text: str) -> List[str]:
        """
        簡單用 jieba 斷詞；也可以改成你喜歡的方式
        """
        return list(jieba.cut(text))

    def _split_sentences(self, text: str) -> List[str]:
        """
        用標點符號切句：。！？；…
        """
        # 先用正則把各種句尾標點都換成「。」，再 split
        tmp = re.sub(r"[！？!？；;]", "。", text)
        sentences = [s.strip() for s in tmp.split("。") if s.strip()]
        return sentences

    def _build_word_freq(self, sentences: List[str]) -> Dict[str, float]:
        """
        全文詞頻（忽略停用詞）
        """
        freq: Dict[str, int] = {}
        for sent in sentences:
            for w in self._tokenize(sent):
                if w in self.stop_words:
                    continue
                if re.match(r"\s+", w):
                    continue
                freq[w] = freq.get(w, 0) + 1

        # 正規化成 0~1
        if not freq:
            return {}
        max_freq = max(freq.values())
        return {w: c / max_freq for w, c in freq.items()}

    def sentence_score(self, sentence: str, word_freq: Dict[str, float]) -> float:
        """
        對單一句子計算分數：把句子內非停用詞的 word_freq 加總，
        再除以句子長度做簡單正規化。
        """
        tokens = self._tokenize(sentence)
        if not tokens:
            return 0.0

        score = 0.0
        useful_len = 0
        for w in tokens:
            if w in self.stop_words:
                continue
            if re.match(r"\s+", w):
                continue
            score += word_freq.get(w, 0.0)
            useful_len += 1

        if useful_len == 0:
            return 0.0
        return score / useful_len

    def summarize(self, text: str, ratio: float = 0.3) -> str:
        """
        統計式摘要：
          1. 切句
          2. 建立全文詞頻
          3. 每句打分
          4. 挑出分數最高的前 top_n 句，依原順序組合成摘要
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return ""

        word_freq = self._build_word_freq(sentences)

        # 每句算分數
        scores = []
        for i, sent in enumerate(sentences):
            s = self.sentence_score(sent, word_freq)
            scores.append((i, s, sent))

        # 依 ratio 決定取幾句（至少 1 句）
        top_n = max(1, int(len(sentences) * ratio))

        # 依分數排序，取前 top_n
        scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

        # 回到原本句子順序
        scores_sorted = sorted(scores_sorted, key=lambda x: x[0])

        summary_sentences = [s[2] for s in scores_sorted]
        return "。".join(summary_sentences) + "。"

# ---------- 測試 main ----------

def main():
    # A-1 測試
    documents = [
        "人工智慧正在改變世界，機器學習是其核心技術",
        "深度學習推動了人工智慧的發展，特別是在圖像識別領域",
        "今天天氣很好，適合出去運動",
        "機器學習和深度學習都是人工智慧的重要分支",
        "運動有益健康，每天都應該保持運動習慣",
    ]

    print("==== 手刻 TF-IDF 相似度矩陣 ====")
    tfidf_dicts = calculate_tfidf(documents)
    sim_manual = cosine_sim_from_dicts(tfidf_dicts)
    print(sim_manual)

    print("\n==== sklearn TF-IDF 相似度矩陣 ====")
    sim_sklearn = sklearn_tfidf_similarity(documents)
    print(sim_sklearn)

    # A-2 測試
    print("\n==== A-2 規則式分類測試 ====")
    test_texts = [
        "這家餐廳的牛肉麵真的太好吃了，湯頭濃郁，麵條Q談，下次一定再來！",
        "最近的AI技術突破讓人驚艷，深度學習模型的表現越來越好",
        "這部電影劇情空洞，演技糟糕，完全是浪費時間",
        "每天慢跑5公里，配合適當的重訓，體能進步很多"
    ]

    sentiment_clf = RuleBasedSentimentClassifier()
    topic_clf = TopicClassifier()

    for i, t in enumerate(test_texts, 1):
        s_label = sentiment_clf.classify(t)
        topic_label = topic_clf.classify(t)
        print(f"[句子{i}] {t}")
        print(f"  -> 情緒: {s_label} / 主題: {topic_label}")

    # ===== A-3 測試：統計式自動摘要 =====
    print("\n==== A-3 統計式摘要測試 ====")

    article = """
    人工智慧(AI)的發展正在深刻改變我們的生活方式。從早上起床時的智慧鬧鐘,
    到通勤時的路線規劃,再到工作中的各種輔助工具,AI無處不在。
    
    在醫療領域,AI協助醫生進行疾病診斷,提高了診斷的準確率和效率。透過分析
    大量的醫療影像和病歷資料,AI能夠發現人眼容易忽略的細節,為患者提供更好的治療方案。
    
    教育方面,AI個人化學習系統能夠根據每個學生的學習進度和特點,提供客製化
    的教學內容。這種因材施教的方式,讓學習變得更加高效和有趣。
    
    然而,AI的快速發展也帶來了一些挑戰。首先是就業問題,許多傳統工作可能會
    被AI取代。其次是隱私和安全問題,AI系統需要大量數據來訓練,如何保護個人
    隱私成為重要議題。最後是倫理問題,AI的決策過程往往缺乏透明度,可能會產
    生偏見或歧視。
    
    面對這些挑戰,我們需要在推動A工發展的同時,建立相應的法律法規和倫理準則。
    只有這樣,才能確保AI技術真正為人類福祉服務,創造一個更美好的未來。

    """

    summarizer = StatisticalSummarizer()
    summary = summarizer.summarize(article, ratio=0.5)

    print("【原文】")
    print(article.strip())
    print("【統計式摘要】")
    clean_summary = summary.replace("\n", "").replace("  ", "")
    print(clean_summary)
    
if __name__ == "__main__":
    main()