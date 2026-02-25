"""Core ReID evaluation engine — metrics, plots, and reporting."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score, roc_curve

from primateid.embedding.multi_embedder import MultiEmbedder
from primateid.evaluation.pairs_generator import generate_pairs, load_pairs

logger = logging.getLogger(__name__)


class ReIDEvaluator:
    """Evaluate re-identification performance using cosine similarity."""

    def __init__(self, embedder: MultiEmbedder) -> None:
        self.embedder = embedder

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def evaluate(self, crops_dir: Path, output_dir: Path) -> dict:
        """Run full evaluation pipeline and write results to *output_dir*."""
        output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        # 1. Pairs
        manual_csv = crops_dir / "pairs.csv"
        if manual_csv.exists():
            logger.info("Using manual pairs from %s", manual_csv)
            pairs = load_pairs(manual_csv)
        else:
            pairs = generate_pairs(crops_dir, output_dir / "pairs.csv")

        # 2. Collect unique images & compute embeddings
        all_images: list[str] = list({p["img1"] for p in pairs} | {p["img2"] for p in pairs})
        all_images.sort()
        logger.info("Computing embeddings for %d images…", len(all_images))

        emb_map: dict[str, np.ndarray] = {}
        batch_size = 32
        for i in range(0, len(all_images), batch_size):
            batch_paths = [Path(p) for p in all_images[i:i + batch_size]]
            vecs = self.embedder.embed_batch(batch_paths)
            for path_str, vec in zip(all_images[i:i + batch_size], vecs):
                emb_map[path_str] = vec

        # Save embeddings
        np.savez(output_dir / "embeddings.npz",
                 paths=np.array(all_images),
                 embeddings=np.stack([emb_map[p] for p in all_images]))

        # 3. Similarities
        labels: list[int] = []
        scores: list[float] = []
        score_rows: list[dict] = []
        for p in pairs:
            sim = float(np.dot(emb_map[p["img1"]], emb_map[p["img2"]]))
            labels.append(int(p["label"]))
            scores.append(sim)
            score_rows.append({
                "img1": p["img1"], "img2": p["img2"],
                "label": p["label"], "similarity": f"{sim:.6f}",
            })

        # Write scores.csv
        import csv
        with open(output_dir / "scores.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["img1", "img2", "label", "similarity"])
            w.writeheader()
            w.writerows(score_rows)

        y_true = np.array(labels)
        y_score = np.array(scores)

        # 4. Metrics
        genuine_scores = y_score[y_true == 1]
        impostor_scores = y_score[y_true == 0]
        metrics = self._compute_metrics(y_true, y_score, genuine_scores, impostor_scores)
        logger.info("AUC=%.4f  EER=%.2f%%  d'=%.3f", metrics["auc"], metrics["eer_pct"], metrics["decidability"])

        # Save summary
        with open(output_dir / "summary.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # 5. Plots
        self._plot_roc(y_true, y_score, metrics, figures_dir / "roc_curve.png")
        self._plot_score_dist(genuine_scores, impostor_scores, metrics, figures_dir / "score_distribution.png")

        # 6. Report
        n_individuals = len({d.name for d in crops_dir.iterdir() if d.is_dir()})
        n_images = len(all_images)
        n_genuine = int(y_true.sum())
        n_impostor = len(y_true) - n_genuine
        self._write_report(output_dir / "report.md", metrics,
                           backbone=self.embedder.backbone_name,
                           crops_dir=str(crops_dir),
                           n_individuals=n_individuals,
                           n_images=n_images,
                           n_genuine=n_genuine,
                           n_impostor=n_impostor)

        return metrics

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_score: np.ndarray,
                         genuine: np.ndarray, impostor: np.ndarray) -> dict:
        auc = float(roc_auc_score(y_true, y_score))

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        fnr = 1 - tpr
        # EER: where FPR ≈ FNR
        eer_func = interp1d(fpr, fnr)
        # Find crossing point
        diff = fpr - fnr
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_changes) > 0:
            idx = sign_changes[0]
            # Linear interpolation
            x1, x2 = fpr[idx], fpr[idx + 1]
            y1, y2 = fnr[idx], fnr[idx + 1]
            d1, d2 = diff[idx], diff[idx + 1]
            alpha = -d1 / (d2 - d1) if d2 != d1 else 0.5
            eer = float(x1 + alpha * (x2 - x1))
            eer_threshold = float(thresholds[idx] + alpha * (thresholds[idx + 1] - thresholds[idx]))
        else:
            eer = float(fpr[np.argmin(np.abs(diff))])
            eer_threshold = float(thresholds[np.argmin(np.abs(diff))])

        # Decidability d'
        mu_g, mu_i = float(genuine.mean()), float(impostor.mean())
        var_g, var_i = float(genuine.var()), float(impostor.var())
        d_prime = (mu_g - mu_i) / np.sqrt(0.5 * (var_g + var_i) + 1e-10)

        # Best threshold: maximise TPR + TNR = (1-FNR) + (1-FPR)
        j_index = tpr + (1 - fpr) - 1  # Youden's J
        best_idx = int(np.argmax(j_index))
        best_threshold = float(thresholds[best_idx])

        return {
            "auc": round(auc, 4),
            "eer": round(eer, 4),
            "eer_pct": round(eer * 100, 2),
            "eer_threshold": round(eer_threshold, 4),
            "decidability": round(float(d_prime), 4),
            "best_threshold": round(best_threshold, 4),
            "genuine_mean": round(mu_g, 4),
            "impostor_mean": round(mu_i, 4),
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    @staticmethod
    def _plot_roc(y_true: np.ndarray, y_score: np.ndarray,
                  metrics: dict, path: Path) -> None:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
        ax.plot(fpr, tpr, color="#2563eb", linewidth=2,
                label=f"ROC (AUC = {metrics['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="#9ca3af", linewidth=1)
        # EER point
        eer = metrics["eer"]
        ax.plot(eer, 1 - eer, "o", color="#dc2626", markersize=8,
                label=f"EER = {metrics['eer_pct']:.1f}%")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve", fontsize=14)
        ax.legend(fontsize=11, loc="lower right")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("ROC curve saved to %s", path)

    @staticmethod
    def _plot_score_dist(genuine: np.ndarray, impostor: np.ndarray,
                         metrics: dict, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
        bins = np.linspace(min(impostor.min(), genuine.min()) - 0.05,
                           max(impostor.max(), genuine.max()) + 0.05, 50)
        ax.hist(genuine, bins=bins, alpha=0.6, color="#3b82f6", label="Genuine", density=True)
        ax.hist(impostor, bins=bins, alpha=0.6, color="#f97316", label="Impostor", density=True)

        # KDE overlay
        try:
            from scipy.stats import gaussian_kde
            xs = np.linspace(bins[0], bins[-1], 200)
            if len(genuine) > 1:
                ax.plot(xs, gaussian_kde(genuine)(xs), color="#1d4ed8", linewidth=2)
            if len(impostor) > 1:
                ax.plot(xs, gaussian_kde(impostor)(xs), color="#c2410c", linewidth=2)
        except Exception:
            pass

        # Threshold line
        ax.axvline(metrics["best_threshold"], color="#16a34a", linestyle="--",
                    linewidth=1.5, label=f"Best threshold = {metrics['best_threshold']:.3f}")
        ax.set_xlabel("Cosine Similarity", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Score Distribution", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Score distribution saved to %s", path)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    @staticmethod
    def _write_report(path: Path, metrics: dict, *,
                      backbone: str, crops_dir: str,
                      n_individuals: int, n_images: int,
                      n_genuine: int, n_impostor: int) -> None:
        md = f"""# PrimateReID Evaluation Report

## Configuration
- **Backbone**: {backbone}
- **Crops directory**: {crops_dir}
- **Individuals**: {n_individuals}
- **Total images**: {n_images}
- **Pairs**: {n_genuine} genuine, {n_impostor} impostor

## Results

| Metric | Value |
|--------|-------|
| AUC | {metrics['auc']:.4f} |
| EER | {metrics['eer_pct']:.1f}% |
| Decidability (d') | {metrics['decidability']:.4f} |
| Best Threshold | {metrics['best_threshold']:.4f} |

## Score Distribution
![Score Distribution](figures/score_distribution.png)

## ROC Curve
![ROC Curve](figures/roc_curve.png)
"""
        path.write_text(md)
        logger.info("Report written to %s", path)
