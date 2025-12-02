# NCCU CS – Homework 2: Traditional NLP Methods vs. Modern AI Methods
---
**Text Similarity · Classification · Summarization**

## 1.🚀 專案概述 (Overview)

本專案實作並比較了兩種主要的自然語言處理 (NLP) 範式：

* **傳統方法 (Part A)**
    * 手動與 `sklearn` TF–IDF 計算
    * 基於規則的情感與主題分類
    * 統計型提取式摘要 (Statistical Extractive Summarization)

* **現代 AI 方法 (Part B)**
    * GPT 嵌入（Embedding）語義相似度
    * GPT-4o-mini 情感與主題分類
    * GPT-4o-mini 抽象式摘要 (Abstractive Summarization)

* **比較流程 (Part C)**
    * 準確度 (Accuracy) 比較
    * 運行時間 (Runtime) 測量
    * 結果視覺化 / 匯出指標

所有實驗結果將自動寫入 `results/` 目錄。

## 2. 📂 專案結構 (Project Structure)

```text
nccu_hw2/
│
├── traditional_methods.py           # Part A (TF–IDF, 規則分類, 統計摘要)
├── modern_methods.py                # Part B (GPT embedding, GPT-4o-mini 分類與摘要)
├── comparison.py                    # Part C (執行所有實驗並匯出指標)
├── report.md                        # 分析報告 (Markdown)
├── requirements.txt                 # Python 依賴項 (用於 pip install)
│
├── results/                         # 自動生成輸出
│   ├── performance_metrics.json     # 相似度、分類、摘要的總體指標
│   ├── classification_results.csv   # 每句文本的標籤與預測結果
│   ├── summarization_comparison.txt # 原始文本 vs. 傳統 vs. GPT 摘要的並列比較
│   └── tfidf_similarity_matrix.png  # (可選) TF–IDF 相似度矩陣的熱力圖視覺化
│
└── README.md                        # 本檔案
```
## 3. 🛠️ 環境設定 (Environment Setup)

### 3.1 Python 版本

建議使用：**Python 3.9+**

### 3.2 建立並啟用虛擬環境

```bash
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
# .\venv\Scripts\activate     # Windows PowerShell
```

### 3.3 安裝依賴項

```bash
pip install -r requirements.txt
```

`requirements.txt` 範例內容： 

```bash
openai==1.57.0
jieba==0.42.1
numpy==2.0.2
scikit-learn==1.5.2
matplotlib==3.9.2
```

## 4. 🔑 OpenAI API Key
Part B 和 Part C 的現代 AI 方法需要一個 OpenAI API key。

在執行腳本之前，請將其設定為環境變數：

```bash
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
```
或將該行新增至 `~/.zshrc` / `~/.bashrc` 並重新載入您的 Shell：

```bash
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
```
## 5. ▶️ 如何執行 (How to Run)
請確保虛擬環境已啟用且 `OPENAI_API_KEY` 已設定。

### 5.1 Part A – 傳統方法

```bash
python traditional_methods.py
```

此腳本將會：
計算一小部分文件集的手動 TF–IDF。
計算相同文件集的 `sklearn` TF–IDF。
列印這兩種餘弦相似度矩陣。

執行：
基於規則的情感分類器
基於規則的主題分類器
為範例文章生成一個統計型提取式摘要。
(TF–IDF 相似度矩陣也可以使用 `matplotlib` 視覺化並儲存為 `results/tfidf_similarity_matrix.png`。)

### 5.2 Part B – 現代 AI 方法
```bash
python modern_methods.py
```
此腳本將會：
使用 OpenAI 嵌入 (`text-embedding-3-small`) 計算句子間的語義相似度。
使用 GPT-4o-mini 進行分類：
情感：`正面` / `負面` / `中性`
主題：`科技` / `運動` / `美食` / `旅遊` / `其他`
信心分數：`0–1`
使用 GPT-4o-mini 根據長度限制生成抽象式摘要。
所有輸出都會記錄到控制台，以便截圖或檢查。

### 5.3 Part C – 比較流程
```bash
python comparison.py
```
此腳本將對三個任務執行小型基準測試：

1. 語義相似度 (Semantic Similarity)
比較： TF–IDF vs. GPT 嵌入
透過閾值將連續相似度轉換為二元標籤
計算準確度與運行時間

2. 文本分類 (Text Classification)
比較： 基於規則 vs. GPT-4o-mini
計算準確度（情感與主題）和運行時間
將每筆樣本的結果匯出至 CSV

3. 摘要 (Summarization)
比較： 統計型摘要器 vs. GPT-4o-mini 摘要器
測量兩者的運行時間
匯出並列摘要，供手動評估

產生的檔案 (皆在 results/ 下)：
`performance_metrics.json`
包含以下指標：
```bash
{
  "similarity": {
    "tfidf": { "accuracy": 0.5, "time_sec": 0.0031, ... },
    "gpt":   { "accuracy": 0.5, "time_sec": 5.3543, ... }
  },
  "classification": {
    "rule_based": { "accuracy_sentiment": 0.8, "accuracy_topic": 0.8, ... },
    "gpt":        { "accuracy_sentiment": 0.8, "accuracy_topic": 0.8, ... }
  },
  "summarization": {
    "statistical": { "time_sec": 0.3060 },
    "gpt":         { "time_sec": 6.0010 }
  }
}
```

`classification_results.csv`

每行 = 一個測試句子，包含：
基本事實 (Ground-truth) 情感 / 主題
基於規則的預測
GPT 預測 + 信心分數

`summarization_comparison.txt`

針對每篇文章：
原始文本
統計型摘要
GPT 摘要
保留空行供手動評分 (信息保留度、流暢度等)

`tfidf_similarity_matrix.png`

TF–IDF 相似度矩陣的可選熱力圖視覺化 。

## 6. 💡 解讀 (Interpretation for Reviewers)

| 特性       | 傳統方法（Traditional Methods）                   | 現代 AI 方法（Modern AI Methods）                     |
|------------|---------------------------------------------------|--------------------------------------------------------|
| 速度       | 快速                                            | 較慢（受 API 延遲影響）                            |
| 成本       | 便宜（無 API 費用）                            | 有 API 費用（成本較高）                            |
| 可解釋性   | 易於解釋（基於規則、統計）                     | 難以解釋（黑箱模型）                               |
| 語義理解   | 較弱（僅基於詞彙共現）                            | 強大（能捕捉深層語義）                            |
| 分類能力   | 較脆弱（依賴手動規則）                            | 更穩健（基於大型模型訓練）                            |
| 摘要品質   | 摘取式，可能不連貫                                | 抽象式，更自然、連貫                                   |

本儲存庫的結構允許每個部分 (A/B/C) 獨立運行，而 `comparison.py` 則將所有內容結合起來，進行定量和定性分析。

## 7. 👤 作者 (Author)
Jack Huang
NCCU CS － Master’s Program (In-Service)
