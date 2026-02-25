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

# 產生範例測試資料
python3 scripts/generate_sample_data.py

# 使用 ResNet50 骨幹網路執行評估
PYTHONPATH=src python3 -m primateid.run --crops data/sample_crops --backbone resnet50

# 或使用 FaceNet 骨幹網路
PYTHONPATH=src python3 -m primateid.run --crops data/sample_crops --backbone facenet
```

### CLI 選項

```
--crops PATH      裁切圖片目錄路徑（必填）
--backbone STR    嵌入骨幹網路：resnet50 | facenet（預設：resnet50）
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

## 目前狀態

**v0.1** — 嵌入 pipeline + 評估 + CLI 已可運作。偵測與裁切模組開發中。

## 相關專案

- [FaceThresholdLab](https://github.com/jonesandjay123/FaceThresholdLab) — 臉部嵌入分析的評估引擎
- [FacialRecognitionTest](https://github.com/jonesandjay123/FacialRecognitionTest) — 早期臉部辨識實驗

## 貢獻者

- **Jones** — 專案負責人、流程架構
- **Eleane（趙以琳）** — SAM3 偵測研究、野外測試

## 授權

MIT
