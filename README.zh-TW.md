# PrimateReID

靈長類臉部偵測、裁切與個體再辨識的端到端流程。

> **[English README](README.md)**

## 架構

```
原始照片 → 偵測 (YOLO/SAM3) → 裁切 (框選/遮罩) → 嵌入向量 (FaceNet/ArcFace/靈長類模型) → 再辨識評估
```

**PrimateReID** 負責從野外原始照片到個體辨識的完整流程。閾值分析與評估指標整合 [FaceThresholdLab](https://github.com/jonesandjay123/FaceThresholdLab) 作為評估引擎。

## 流程元件

### 偵測（Detection）
使用 YOLO 或 SAM3 進行臉部／身體偵測前端，在原始照片中定位靈長類目標。

### 裁切（Cropping）
透過邊界框裁切或遮罩裁切提取個體區域，為嵌入模型準備乾淨的輸入。

### 嵌入（Embedding）
使用多種骨幹網路生成具身份鑑別力的特徵向量——FaceNet、ArcFace 或靈長類專用模型。

### 評估（Evaluation）
將嵌入向量送入 [FaceThresholdLab](https://github.com/jonesandjay123/FaceThresholdLab) 進行距離分析、閾值調整與再辨識準確率報告。

## 目前狀態

**早期開發** — 第一階段偵測比較（SAM3 vs YOLO）已完成，流程整合進行中。

### 第一階段：SAM3 偵測結果

E小姐針對三種場景進行了 SAM3 零樣本偵測實驗：

| 場景 | 條件 | 發現 |
|------|------|------|
| S1 | 單隻猴子，乾淨背景 | 分割品質良好 |
| S2 | 多隻猴子，中度遮擋 | 偵測尚可，部分臉部遺漏 |
| S3 | 野外條件（混合物種、雜亂背景） | 僅靠提示工程不足以達到可靠偵測 |

這些結果促成了採用多階段流程搭配專用偵測器的決策。完整探索內容保存於 [`archive/sam3-exploration/`](archive/sam3-exploration/)。

## 專案結構

```
PrimateReID/
├── src/primateid/        # 核心流程模組
│   ├── detection/        # YOLO、SAM3 偵測前端
│   ├── cropping/         # 框選裁切、遮罩裁切
│   ├── embedding/        # FaceNet、ArcFace、靈長類模型
│   ├── evaluation/       # FaceThresholdLab 整合
│   └── utils/
├── configs/              # 實驗設定（YAML）
├── data/                 # 測試資料（已 gitignore）
├── results/              # 實驗結果輸出
├── archive/              # 第一階段 SAM3 探索（已保留）
└── tests/
```

## 快速開始

```bash
git clone https://github.com/jonesandjay123/PrimateReID.git
cd PrimateReID
pip install -r requirements.txt
# 流程使用方式——即將推出
```

## 相關專案

- [FaceThresholdLab](https://github.com/jonesandjay123/FaceThresholdLab) — 臉部嵌入分析的評估引擎
- [FacialRecognitionTest](https://github.com/jonesandjay123/FacialRecognitionTest) — 早期臉部辨識實驗

## 貢獻者

- **Jones** — 專案負責人、流程架構
- **Eleane（趙以琳）** — SAM3 偵測研究、野外測試

## 授權

MIT
