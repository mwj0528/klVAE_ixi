import argparse
import json
import os
import shutil
import sys

import nibabel as nib
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data.dataloader import default_collate
from tqdm import trange

sys.path.append(os.getcwd())

from main import instantiate_from_config
from metrics import (
    LATENT_SUMMARY_KEYS,
    RECON_SUMMARY_KEYS,
    aggregate_modality_summaries,
    compute_metrics,
    evaluate_latent_for_diffusion,
    save_metrics_csv,
    save_metrics_json,
    save_modality_aggregate,
    summarize_metrics,
)


def get_parser():
    parser = argparse.ArgumentParser(description="VQ-GAN stage1 reconstruction inference")
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        required=True,
        help="Checkpoint path (.ckpt)",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        default=list(),
        help="Config yaml path(s)",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default=None,
        choices=["t1", "t2", "pd", "t1ce", "flair"],
        help="Single modality to reconstruct (use --modalities for multiple)",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        choices=["t1", "t2", "pd", "t1ce", "flair"],
        help="Multiple modalities to reconstruct and aggregate",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to run on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Limit number of samples (-1 for all)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run on CUDA",
    )
    parser.add_argument(
        "--data_range",
        type=float,
        default=1.0,
        help="Value range for PSNR/SSIM (ADNI preprocessing uses [0, 1])",
    )
    return parser


def load_model_from_config(config, sd, gpu=True):
    if "ckpt_path" in config.params:
        config.params.ckpt_path = None
    model = instantiate_from_config(config)
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
    if gpu and torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


def get_dataset(config, split, modality):
    split_key = {
        "train": "train",
        "val": "validation",
        "test": "test",
    }[split]

    data_cfg = OmegaConf.to_container(config.data, resolve=True)
    params = dict(data_cfg.get("params") or {})
    data_path = params.pop("data_path", None)
    split_cfg = params.get(split_key)
    if split_cfg is None:
        raise ValueError(f"Split '{split_key}' not found in data config")

    if not isinstance(split_cfg, dict):
        split_cfg = OmegaConf.to_container(split_cfg, resolve=True)
    ds_params = dict(split_cfg.get("params") or {})
    if data_path is not None:
        ds_params.setdefault("data_path", data_path)
    ds_params.setdefault("source", modality)
    ds_params.setdefault("target", modality)

    dataset = instantiate_from_config(
        {
            "target": split_cfg["target"],
            "params": ds_params,
        }
    )
    return dataset


def tensor_to_nifti(x):
    x = x.detach().cpu()
    if x.ndim == 5:
        x = x[0]
    if x.ndim == 4:
        x = x[0]
    return x.numpy()


def save_nifti(volume, path):
    volume = np.asarray(volume)
    if volume.ndim == 3:
        volume = volume[..., None]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(volume, np.eye(4)), path)


def prepare_output_dirs(modality_outdir):
    gt_dir = os.path.join(modality_outdir, "GT")
    pred_dir = os.path.join(modality_outdir, "pred")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    return gt_dir, pred_dir


def copy_checkpoint_to_outdir(source_ckpt, outdir):
    os.makedirs(outdir, exist_ok=True)
    ckpt_dir = os.path.join(outdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_name = os.path.basename(source_ckpt)
    local_ckpt = os.path.join(ckpt_dir, ckpt_name)
    source_ckpt = os.path.abspath(source_ckpt)
    local_ckpt = os.path.abspath(local_ckpt)

    if source_ckpt != local_ckpt:
        print(f"Copying checkpoint to: {local_ckpt}")
        shutil.copy2(source_ckpt, local_ckpt)

    run_info = {
        "source_checkpoint": source_ckpt,
        "local_checkpoint": local_ckpt,
    }
    with open(os.path.join(outdir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    return local_ckpt


@torch.no_grad()
def run_reconstruction(
    model,
    dataset,
    outdir,
    modality,
    batch_size=1,
    max_samples=-1,
    gpu=True,
    data_range=1.0,
):
    os.makedirs(outdir, exist_ok=True)
    gt_dir, pred_dir = prepare_output_dirs(outdir)
    n_samples = len(dataset)
    if max_samples > 0:
        n_samples = min(n_samples, max_samples)

    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    metric_rows = []

    for start_idx in trange(0, n_samples - batch_size + 1, batch_size):
        indices = list(range(start_idx, start_idx + batch_size))
        example = default_collate([dataset[i] for i in indices])
        subject_id = example.get("subject_id", [f"sample_{start_idx:04d}"])[0]

        x = model.get_input(example, modality).to(device)
        z, _, info = model.encode(x)
        recon = model.decode(z)

        input_vol = tensor_to_nifti(x)
        recon_vol = tensor_to_nifti(recon)

        sample_metrics = compute_metrics(
            recon_vol,
            input_vol,
            data_range=data_range,
            clip_pred=True,
        )
        code_indices = info[2].detach().cpu().numpy()
        latent_metrics = evaluate_latent_for_diffusion(
            z.detach().cpu().numpy(),
            code_indices,
            n_embed=model.quantize.n_e,
        )
        metric_rows.append(
            {
                "subject_id": subject_id,
                "modality": modality,
                **sample_metrics,
                **latent_metrics,
            }
        )

        gt_path = os.path.join(gt_dir, f"{subject_id}.nii.gz")
        pred_path = os.path.join(pred_dir, f"{subject_id}.nii.gz")
        save_nifti(input_vol, gt_path)
        save_nifti(recon_vol, pred_path)

    summary = summarize_metrics(
        metric_rows,
        keys=RECON_SUMMARY_KEYS + LATENT_SUMMARY_KEYS,
    )
    summary["modality"] = modality
    summary["data_range"] = data_range
    summary["n_embed"] = model.quantize.n_e

    save_metrics_csv(metric_rows, os.path.join(outdir, "metrics.csv"))
    save_metrics_json(summary, os.path.join(outdir, "metrics_summary.json"))

    print("\n=== Reconstruction Metrics ===")
    print(f"n={summary.get('n', 0)}")
    if "psnr" in summary:
        print(f"PSNR: {summary['psnr']['mean']:.4f} +/- {summary['psnr']['std']:.4f}")
    if "ssim" in summary:
        print(f"SSIM: {summary['ssim']['mean']:.4f} +/- {summary['ssim']['std']:.4f}")
    if "hfen" in summary:
        print(f"HFEN: {summary['hfen']['mean']:.4f} +/- {summary['hfen']['std']:.4f} (lower=better)")
    if "gmsd" in summary:
        print(f"GMSD: {summary['gmsd']['mean']:.4f} +/- {summary['gmsd']['std']:.4f} (lower=better)")

    print("\n=== Latent / Diffusion Readiness ===")
    if "latent_std" in summary:
        print(f"latent_std: {summary['latent_std']['mean']:.4f} +/- {summary['latent_std']['std']:.4f}")
    if "scale_factor" in summary:
        print(f"scale_factor (1/std): {summary['scale_factor']['mean']:.4f} +/- {summary['scale_factor']['std']:.4f}")
    if "codebook_active_ratio" in summary:
        print(
            f"codebook_active_ratio: {summary['codebook_active_ratio']['mean']:.4f} "
            f"+/- {summary['codebook_active_ratio']['std']:.4f}"
        )
    if "codebook_perplexity" in summary:
        print(
            f"codebook_perplexity: {summary['codebook_perplexity']['mean']:.1f} "
            f"+/- {summary['codebook_perplexity']['std']:.1f} "
            f"(n_embed={model.quantize.n_e})"
        )
    print(f"Saved: {os.path.join(outdir, 'metrics.csv')}")
    print(f"Saved: {os.path.join(outdir, 'metrics_summary.json')}")

    return metric_rows, summary


def resolve_modalities(modality, modalities):
    if modalities:
        return modalities
    if modality:
        return [modality]
    return ["t1"]


def aggregate_and_print_summaries(outdir, modality_summaries):
    aggregate = aggregate_modality_summaries(modality_summaries)
    save_modality_aggregate(aggregate, outdir)

    print("\n=== Reconstruction Summary (per modality) ===")
    for row in aggregate["modalities"]:
        print(
            f"{row['modality']}: "
            f"PSNR={row['psnr_mean']:.4f} +/- {row['psnr_std']:.4f}, "
            f"SSIM={row['ssim_mean']:.4f} +/- {row['ssim_std']:.4f}, "
            f"HFEN={row['hfen_mean']:.4f}, GMSD={row['gmsd_mean']:.4f} "
            f"(n={row['n']})"
        )
        print(
            f"  latent_std={row['latent_std_mean']:.4f}, "
            f"scale_factor={row['scale_factor_mean']:.4f}, "
            f"codebook_active_ratio={row['codebook_active_ratio_mean']:.4f}, "
            f"codebook_perplexity={row['codebook_perplexity_mean']:.1f}"
        )

    overall = aggregate["overall"]
    latent_overall = aggregate.get("latent_overall", {})
    print("\n=== Overall (mean of modality means) ===")
    print(f"PSNR: {overall['psnr']['mean']:.4f} +/- {overall['psnr']['std']:.4f}")
    print(f"SSIM: {overall['ssim']['mean']:.4f} +/- {overall['ssim']['std']:.4f}")
    if "hfen" in overall:
        print(f"HFEN: {overall['hfen']['mean']:.4f} +/- {overall['hfen']['std']:.4f}")
    if "gmsd" in overall:
        print(f"GMSD: {overall['gmsd']['mean']:.4f} +/- {overall['gmsd']['std']:.4f}")
    if latent_overall:
        print(
            f"latent_std: {latent_overall['latent_std']['mean']:.4f}, "
            f"scale_factor: {latent_overall['scale_factor']['mean']:.4f}, "
            f"codebook_active_ratio: {latent_overall['codebook_active_ratio']['mean']:.4f}, "
            f"codebook_perplexity: {latent_overall['codebook_perplexity']['mean']:.1f}"
        )
    print(f"Saved: {os.path.join(outdir, 'all_modalities_summary.json')}")
    print(f"Saved: {os.path.join(outdir, 'all_modalities_summary.csv')}")
    return aggregate


if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    if not opt.base:
        raise ValueError("Please provide config with -b configs/adni_vqgan_stage1.yaml")

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    if not os.path.isfile(opt.resume):
        raise FileNotFoundError(f"Checkpoint not found: {opt.resume}")

    source_ckpt = opt.resume
    os.makedirs(opt.outdir, exist_ok=True)
    local_ckpt = copy_checkpoint_to_outdir(source_ckpt, opt.outdir)

    pl_sd = torch.load(local_ckpt, map_location="cpu")
    model = load_model_from_config(config.model, pl_sd["state_dict"], gpu=opt.gpu)

    modalities = resolve_modalities(opt.modality, opt.modalities)
    modality_summaries = {}

    print(f"Source checkpoint: {source_ckpt}")
    print(f"Using checkpoint: {local_ckpt}")
    print(f"Stage: {getattr(model, 'stage', 'unknown')}")
    print(f"Modalities: {modalities}")
    print(f"Writing outputs to: {opt.outdir}")

    for modality in modalities:
        mod_outdir = os.path.join(opt.outdir, modality)
        dataset = get_dataset(config, opt.split, modality)

        print(f"\n=== Running reconstruction: {modality} ===")
        print(f"Dataset: {dataset.__class__.__name__}, n={len(dataset)}")

        _, summary = run_reconstruction(
            model,
            dataset,
            mod_outdir,
            modality,
            batch_size=opt.batch_size,
            max_samples=opt.max_samples,
            gpu=opt.gpu,
            data_range=opt.data_range,
        )
        modality_summaries[modality] = summary

    if len(modalities) > 1:
        aggregate_and_print_summaries(opt.outdir, modality_summaries)
