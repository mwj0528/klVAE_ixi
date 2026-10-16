#!/usr/bin/env python3
"""Build IXI_datalist_3D.json (same layout as ADNI_datalist_3D.json).

Expects {root}/{train,val}/{t1,t2,pd}/{subject}.nii.gz.
Volume stats are computed over foreground voxels (> 0; IXI is skull-stripped).
As in ADNI, "test" is a copy of "validation".
"""
import argparse
import json
import os
from multiprocessing import Pool

import nibabel as nib
import numpy as np

MODALITIES = ["t1", "t2", "pd"]
SPLITS = {"train": "training", "val": "validation"}


def volume_stats(path):
    data = np.asanyarray(nib.load(path).dataobj)
    fg = data[data > 0]
    return tuple(data.shape), {
        "mean": float(fg.mean()), "std": float(fg.std()), "median": float(np.median(fg)),
        "min": float(fg.min()), "max": float(fg.max()),
    }


def build_entry(args):
    root, split, subject = args
    entry, shapes = {}, {}
    for m in MODALITIES:
        entry[m] = f"{m}/{subject}.nii.gz"
    for m in MODALITIES:
        shapes[m], stats = volume_stats(os.path.join(root, split, entry[m]))
        for k, v in stats.items():
            entry[f"{m}_volume_{k}"] = v
    entry.update({"subject_id": subject, "split": split})
    return entry, shapes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch2/mailab_yonsei/MW/lustre/MRI/IXI/3D")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min_slices", type=int, default=100, help="drop subjects with any axis shorter than this")
    a = ap.parse_args()

    out = {
        "description": "IXI",
        "modality": {"0": "T1w", "1": "T2w", "2": "PDw"},
        "tensorImageSize": "3D",
        "volumeStatsSource": "3D",
        "volumeStats3DRoot": a.root,
        "volumeStatsRegion": "foreground_gt0",
        "imageLayout": "{root_3d}/{split}/{modality}/{subject_id}.nii.gz",
        "testEqualsValidation": True,
    }
    for split, key in SPLITS.items():
        sd = os.path.join(a.root, split)
        subjects = {m: {f[: -len(".nii.gz")] for f in os.listdir(os.path.join(sd, m))} for m in MODALITIES}
        common = sorted(set.intersection(*subjects.values()))
        for m in MODALITIES:
            if subjects[m] != set(common):
                print(f"[{split}] WARNING: {m} subjects not in all modalities: {sorted(subjects[m] - set(common))}")
        with Pool(a.workers) as p:
            res = p.map(build_entry, [(a.root, split, s) for s in common])
        entries = []
        for e, shapes in res:
            if len({tuple(s) for s in shapes.values()}) != 1:
                print(f"[{split}] DROP {e['subject_id']}: shape mismatch {shapes}")
            elif min(next(iter(shapes.values()))) < a.min_slices:
                print(f"[{split}] DROP {e['subject_id']}: small volume {shapes['t1']}")
            else:
                entries.append(e)
        out[key] = entries
        print(f"[{split}] {len(entries)}/{len(common)} subjects")

    out["test"] = out["validation"]
    out["numTraining"], out["numValidation"], out["numTest"] = (len(out[k]) for k in ("training", "validation", "test"))
    path = os.path.join(a.root, "IXI_datalist_3D.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
