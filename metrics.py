import json
import os
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity


ArrayLike = Union[np.ndarray, float]

RECON_SUMMARY_KEYS = ["psnr", "ssim", "hfen", "gmsd"]
LATENT_SUMMARY_KEYS = [
    "latent_std",
    "scale_factor",
    "latent_mean_abs",
    "codebook_active_ratio",
    "codebook_perplexity",
]


def to_numpy_volume(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float64)
    if volume.ndim == 4 and volume.shape[-1] == 1:
        volume = volume[..., 0]
    while volume.ndim > 3:
        volume = volume[0]
    if volume.ndim == 2:
        volume = volume[..., None]
    return volume


def to_numpy_volume_3d(volume: np.ndarray) -> np.ndarray:
    volume = to_numpy_volume(volume)
    if volume.ndim == 4 and volume.shape[-1] == 1:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")
    return volume


def highpass_3d(volume: np.ndarray) -> np.ndarray:
    volume = to_numpy_volume_3d(volume)
    return ndimage.laplace(volume)


def gradient_magnitude_3d(volume: np.ndarray) -> np.ndarray:
    volume = to_numpy_volume_3d(volume)
    gradients = [ndimage.sobel(volume, axis=axis) for axis in range(3)]
    return np.sqrt(sum(grad ** 2 for grad in gradients))


def compute_hfen(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-10,
) -> float:
    pred = to_numpy_volume_3d(pred)
    target = to_numpy_volume_3d(target)

    h_target = highpass_3d(target)
    h_pred = highpass_3d(pred)
    denom = np.linalg.norm(h_target.ravel())
    if denom < eps:
        return 0.0
    return float(np.linalg.norm((h_target - h_pred).ravel()) / denom)


def compute_gmsd(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float = 1.0,
    eps: float = 1e-10,
) -> float:
    pred = to_numpy_volume_3d(pred)
    target = to_numpy_volume_3d(target)

    grad_target = gradient_magnitude_3d(target)
    grad_pred = gradient_magnitude_3d(pred)
    c = (0.01 * data_range) ** 2

    gms_map = (2.0 * grad_target * grad_pred + c) / (
        grad_target ** 2 + grad_pred ** 2 + c + eps
    )
    return float(np.sqrt(np.mean((gms_map - np.mean(gms_map)) ** 2)))


def compute_psnr(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float = 1.0,
    eps: float = 1e-10,
) -> float:
    pred = to_numpy_volume_3d(pred)
    target = to_numpy_volume_3d(target)
    mse = np.mean((pred - target) ** 2)
    if mse < eps:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def compute_ssim(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float = 1.0,
) -> float:
    pred = to_numpy_volume_3d(pred)
    target = to_numpy_volume_3d(target)
    return float(
        structural_similarity(
            target,
            pred,
            data_range=data_range,
        )
    )


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float = 1.0,
    clip_pred: bool = True,
) -> Dict[str, float]:
    pred = to_numpy_volume_3d(pred)
    target = to_numpy_volume_3d(target)

    if clip_pred:
        pred = np.clip(pred, 0.0, data_range)

    return {
        "psnr": compute_psnr(pred, target, data_range=data_range),
        "ssim": compute_ssim(pred, target, data_range=data_range),
        "hfen": compute_hfen(pred, target),
        "gmsd": compute_gmsd(pred, target, data_range=data_range),
    }


def to_numpy_latent(latent: np.ndarray) -> np.ndarray:
    latent = np.asarray(latent, dtype=np.float64)
    while latent.ndim > 4:
        latent = latent[0]
    return latent


def compute_codebook_utilization(
    code_indices: np.ndarray,
    n_embed: int,
    eps: float = 1e-10,
) -> Dict[str, float]:
    indices = np.asarray(code_indices, dtype=np.int64).reshape(-1)
    counts = np.bincount(indices, minlength=n_embed).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return {
            "codebook_active_codes": 0.0,
            "codebook_active_ratio": 0.0,
            "codebook_perplexity": 0.0,
            "codebook_entropy": 0.0,
        }

    probs = counts / total
    active_probs = probs[probs > 0]
    entropy = float(-(active_probs * np.log(active_probs + eps)).sum())
    perplexity = float(np.exp(entropy))
    active_codes = float((counts > 0).sum())

    return {
        "codebook_active_codes": active_codes,
        "codebook_active_ratio": active_codes / float(n_embed),
        "codebook_perplexity": perplexity,
        "codebook_entropy": entropy,
    }


def compute_latent_distribution_stats(
    latent: np.ndarray,
    eps: float = 1e-10,
) -> Dict[str, float]:
    latent = to_numpy_latent(latent)
    std_global = float(latent.std())
    mean_global = float(latent.mean())
    mean_abs = float(np.mean(np.abs(latent)))

    per_channel_std = latent.std(axis=tuple(range(1, latent.ndim)))
    per_channel_mean = latent.mean(axis=tuple(range(1, latent.ndim)))

    return {
        "latent_mean": mean_global,
        "latent_std": std_global,
        "latent_mean_abs": mean_abs,
        "latent_min": float(latent.min()),
        "latent_max": float(latent.max()),
        "latent_channel_std_mean": float(per_channel_std.mean()),
        "latent_channel_std_std": float(per_channel_std.std(ddof=0)),
        "latent_channel_mean_std": float(per_channel_mean.std(ddof=0)),
        "scale_factor": float(1.0 / (std_global + eps)),
    }


def evaluate_latent_for_diffusion(
    latent: np.ndarray,
    code_indices: np.ndarray,
    n_embed: int,
) -> Dict[str, float]:
    stats = compute_latent_distribution_stats(latent)
    codebook = compute_codebook_utilization(code_indices, n_embed)
    return {**stats, **codebook}


def summarize_metrics(rows: Iterable[Dict[str, ArrayLike]], keys: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    rows = list(rows)
    if not rows:
        return {}

    if keys is None:
        keys = ["psnr", "ssim"]

    summary = {"n": len(rows)}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            summary[key] = {"mean": float("nan"), "std": float("nan")}
        else:
            summary[key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
            }
    return summary


def save_metrics_csv(rows: List[Dict[str, ArrayLike]], path: str) -> None:
    import csv

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_metrics_json(summary: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def aggregate_modality_summaries(modality_summaries: Dict[str, Dict]) -> Dict:
    rows = []
    for modality, summary in modality_summaries.items():
        row = {
            "modality": modality,
            "n": summary.get("n", 0),
            "psnr_mean": summary.get("psnr", {}).get("mean", float("nan")),
            "psnr_std": summary.get("psnr", {}).get("std", float("nan")),
            "ssim_mean": summary.get("ssim", {}).get("mean", float("nan")),
            "ssim_std": summary.get("ssim", {}).get("std", float("nan")),
        }
        for key in RECON_SUMMARY_KEYS:
            if key in ("psnr", "ssim"):
                continue
            row[f"{key}_mean"] = summary.get(key, {}).get("mean", float("nan"))
            row[f"{key}_std"] = summary.get(key, {}).get("std", float("nan"))
        for key in LATENT_SUMMARY_KEYS:
            row[f"{key}_mean"] = summary.get(key, {}).get("mean", float("nan"))
            row[f"{key}_std"] = summary.get(key, {}).get("std", float("nan"))
        rows.append(row)

    overall = summarize_metrics(
        [
            {
                "psnr": row["psnr_mean"],
                "ssim": row["ssim_mean"],
                "hfen": row["hfen_mean"],
                "gmsd": row["gmsd_mean"],
            }
            for row in rows
        ],
        keys=RECON_SUMMARY_KEYS,
    )
    latent_overall = summarize_metrics(
        [{key: row[f"{key}_mean"] for key in LATENT_SUMMARY_KEYS} for row in rows],
        keys=LATENT_SUMMARY_KEYS,
    )

    return {
        "modalities": rows,
        "overall": overall,
        "latent_overall": latent_overall,
        "diffusion_readiness_notes": {
            "latent_std": "LDM rescales latents by 1/std; stable std across samples is desirable.",
            "scale_factor": "Suggested LDM scale_factor (=1/latent_std). Similar values across modalities help training.",
            "codebook_active_ratio": "Higher is better; very low values indicate codebook collapse.",
            "codebook_perplexity": "Higher indicates more uniform code usage; compare to n_embed.",
            "latent_mean_abs": "Closer to 0 is generally better for diffusion noise schedule assumptions.",
            "hfen": "Lower is better. High-frequency error; sensitive to smoothing/blur.",
            "gmsd": "Lower is better. Gradient magnitude similarity deviation; sensitive to edge smoothing.",
        },
    }


def save_modality_aggregate(summary: Dict, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    save_metrics_json(summary, os.path.join(outdir, "all_modalities_summary.json"))
    save_metrics_csv(summary["modalities"], os.path.join(outdir, "all_modalities_summary.csv"))
