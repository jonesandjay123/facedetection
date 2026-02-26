# PrimateReID

靈長類臉部偵測、裁切與個體再辨識的端到端流程。

> **[English README](README.md)**

## 架構

```
原始照片 → 偵測 (YOLO/SAM3) → 裁切 (框選/遮罩) → 嵌入向量 (FaceNet/ArcFace/靈長類模型) → 再辨識評估
```

**PrimateReID** 負責從野外原始照片到個體辨識的完整流程，內建評估指標（AUC、EER、可分辨度）與視覺化圖表。

## 快速開始

```bash
git clone https://github.com/jonesandjay123/PrimateReID.git
cd PrimateReID
pip install -r requirements.txt
```

### 方案 A：使用內附的真實黑猩猩測試資料

Repo 內附 `data/demo_chimp_crops/` — 10 隻黑猩猩 × 30 張裁切臉部照片（共 300 張，約 22MB），來自 CTai/CZoo 資料集，clone 後即可直接測試：

```bash
# ResNet50 — ImageNet 通用模型（2048 維）
PYTHONPATH=src python3 -m primateid.run --crops data/demo_chimp_crops --backbone resnet50

# FaceNet — 人臉專用模型（512 維）
PYTHONPATH=src python3 -m primateid.run --crops data/demo_chimp_crops --backbone facenet

# ArcFace — SOTA 人臉辨識，InsightFace（512 維）
PYTHONPATH=src python3 -m primateid.run --crops data/demo_chimp_crops --backbone arcface

# DINOv2 — 自監督視覺特徵（384 維）⭐ 表現最佳
PYTHONPATH=src python3 -m primateid.run --crops data/demo_chimp_crops --backbone dinov2
```

### 方案 B：使用合成測試資料

```bash
# 產生隨機測試圖片（不需額外下載）
python3 scripts/generate_sample_data.py

PYTHONPATH=src python3 -m primateid.run --crops data/sample_crops --backbone resnet50
```

### 方案 C：使用自己的資料

將裁切好的圖片放進 `data/你的資料集/<個體名稱>/` 資料夾（參考下方[資料格式](#資料格式)），然後：

```bash
PYTHONPATH=src python3 -m primateid.run --crops data/your_dataset --backbone resnet50
```

### CLI 選項

```
--crops PATH      裁切圖片目錄路徑（必填）
--backbone STR    嵌入骨幹網路：resnet50 | facenet | arcface | dinov2（預設：resnet50）
--output PATH     輸出目錄（預設：results/<backbone>_<timestamp>/）
--device STR      Torch 裝置（預設：cpu）
```

### 輸出結構

```
results/resnet50_20260225_173000/
├── config.json              # 執行參數
├── pairs.csv                # 評估使用的配對
├── embeddings.npz           # 所有嵌入向量（方便重跑評估不用重算）
├── scores.csv               # img1, img2, label, similarity
├── summary.json             # AUC, EER, d', threshold
├── figures/
│   ├── roc_curve.png        # ROC 曲線（標示 EER 點）
│   └── score_distribution.png  # 同一人 vs 不同人分數分佈直方圖
└── report.md                # 人類可讀的摘要報告
```

### 資料格式

將裁切圖片按個體身份放入子資料夾：

```
data/crops/
├── monkey_A/
│   ├── 001.jpg
│   └── 002.jpg
├── monkey_B/
│   └── 001.jpg
```

配對會自動從資料夾結構產生（同資料夾=genuine、不同資料夾=impostor）。若要使用自訂配對，請在 crops 目錄中放置 `pairs.csv`。

## 流程元件

### 偵測（Detection）
使用 YOLO 或 SAM3 進行臉部／身體偵測前端，在原始照片中定位靈長類目標。

### 裁切（Cropping）
透過邊界框裁切或遮罩裁切提取個體區域，為嵌入模型準備乾淨的輸入。

### 嵌入（Embedding）
使用多種骨幹網路生成具身份鑑別力的特徵向量：
- **ResNet50** — ImageNet 預訓練，2048 維嵌入
- **FaceNet** — VGGFace2 預訓練 InceptionResNetV1，512 維嵌入
- **ArcFace** — InsightFace buffalo_l（MS1MV2），512 維嵌入，角度間距損失
- **DINOv2** — Meta 自監督 ViT-S/14，384 維嵌入（不需標籤）

所有嵌入向量皆經 L2 正規化，使 cosine similarity = 內積。

### 評估（Evaluation）
內建評估引擎，計算：
- **AUC** — ROC 曲線下面積
- **EER** — 等錯誤率
- **d'（可分辨度）** — genuine 與 impostor 分佈的分離程度
- **Best threshold** — 最佳操作點（Youden's J）

## 專案結構

```
PrimateReID/
├── src/primateid/        # 核心流程模組
│   ├── detection/        # YOLO、SAM3 偵測前端
│   ├── cropping/         # 框選裁切、遮罩裁切
│   ├── embedding/        # 多骨幹嵌入器
│   ├── evaluation/       # 配對生成 + 指標 + 繪圖
│   └── utils/
├── scripts/              # 工具腳本
├── configs/              # 實驗設定（YAML）
├── data/                 # 測試資料
├── results/              # 實驗結果輸出
└── tests/
```

## 測試資料

Repo 內附兩組測試資料：

| 資料集 | 路徑 | 說明 |
|--------|------|------|
| **黑猩猩 Demo** | `data/demo_chimp_crops/` | 10 隻個體 × 30 張真實臉部裁切，來自 CTai/CZoo（約 22MB） |
| **合成範例** | `data/sample_crops/` | 由 `scripts/generate_sample_data.py` 產生（隨機雜訊，用於 CI/冒煙測試） |

黑猩猩 Demo 資料讓你 clone 後馬上可以跑，不需要額外下載任何東西。

## 目前狀態

**v0.1** — 嵌入 pipeline + 評估 + CLI 已可運作。偵測與裁切模組開發中。

### 基線結果（2026-02-25）

在 `data/demo_chimp_crops/` 上測試 — 10 隻黑猩猩 × 30 張臉部裁切（300 張）：

| 骨幹網路 | 訓練方式 | AUC | EER | d' | 評語 |
|----------|---------|-----|-----|----|------|
| **DINOv2** ⭐ | 自監督（LVD-142M）| **0.725** | **34.7%** | **0.80** | 最佳 — 不需標籤 |
| ResNet50 | 監督式（ImageNet）| 0.688 | 36.3% | 0.67 | 通用特徵，尚可 |
| FaceNet | 人臉（VGGFace2）| 0.614 | 42.2% | 0.41 | 對人臉過度專精 |
| ArcFace | 人臉（MS1MV2）| 0.551 | 45.4% | 0.16 | 太偏人臉，接近亂猜 |

**核心發現**：人臉專用模型（FaceNet、ArcFace）在靈長類臉上表現*更差*。自監督學習（DINOv2）跨物種泛化最佳 — 學到的視覺特徵沒有人類特定偏差。

完整分析、更多候選骨幹與下一步計畫見 [docs/baseline-results.zh-TW.md](docs/baseline-results.zh-TW.md)。

## 相關專案

- [FaceThresholdLab](https://github.com/jonesandjay123/FaceThresholdLab) — 臉部嵌入分析的評估引擎
- [FacialRecognitionTest](https://github.com/jonesandjay123/FacialRecognitionTest) — 早期臉部辨識實驗

## 貢獻者

- **Jones** — 專案負責人、流程架構
- **Eleane（趙以琳）** — SAM3 偵測研究、野外測試

## 授權

MIT
