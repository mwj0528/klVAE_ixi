import os

import numpy as np
import torch
from monai import transforms
from monai.data import Dataset as MonaiDataset, load_decathlon_datalist
from torch.utils.data import Dataset

SPLIT_TO_JSON_KEY = {
    "train": "training",
    "val": "validation",
    "test": "test",
}

DATASET_CONFIG = {
    "brats": {
        "keys": ["t1", "t2", "t1ce", "flair"],
        "split_3d_map": {"train": "training", "val": "val", "test": "val"},
    },
    "adni": {
        "keys": ["t1", "t2", "pd"],
        "split_3d_map": {"train": "train", "val": "val", "test": "val"},
    },
    "ixi": {
        "keys": ["t1", "t2", "pd"],
        "split_3d_map": {"train": "train", "val": "val", "test": "val"},
    }
}


class NormalizeZero:
    def __init__(self):
        pass

    def __call__(self, data):
        data[data<0] = 0
        return data

brats_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["t1", "t1ce", "t2", "flair"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair"], allow_missing_keys=True),
        transforms.Lambdad(keys=["t1", "t1ce", "t2", "flair"], func=lambda x: x[0, :, :, :]),
        transforms.EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair"], allow_missing_keys=True),
        transforms.EnsureTyped(keys=["t1", "t1ce", "t2", "flair"]),
        transforms.Orientationd(keys=["t1", "t1ce", "t2", "flair"], axcodes="RAI", allow_missing_keys=True),
        transforms.Resized(keys=["t1", "t1ce", "t2", "flair"], spatial_size=(128, 128, 128), allow_missing_keys=True),
        transforms.CropForegroundd(keys=["t1", "t1ce", "t2", "flair"], source_key="t1", allow_missing_keys=True),
        transforms.SpatialPadd(keys=["t1", "t1ce", "t2", "flair"], spatial_size=(128, 128, 128), allow_missing_keys=True),
        # transforms.RandSpatialCropd( keys=["t1", "t1ce", "t2", "flair"],
        #     roi_size=(128, 128, 128),
        #     random_center=True, 
        #     random_size=False,
        # ),
        transforms.Lambdad(keys=["t1", "t1ce", "t2", "flair"], func=NormalizeZero()),
        transforms.ScaleIntensityRangePercentilesd(keys=["t1", "t1ce", "t2", "flair"], lower=0, upper=99.5, b_min=0, b_max=1),
        transforms.ToTensord(keys=["t1", "t1ce", "t2", "flair"]),
    ]
)


adni_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["t1","t2", "pd"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["t1","t2", "pd"], allow_missing_keys=True),
        transforms.Lambdad(keys=["t1","t2", "pd"], func=lambda x: x[0, :, :, :]),
        transforms.EnsureChannelFirstd(keys=["t1","t2", "pd"], allow_missing_keys=True),
        transforms.EnsureTyped(keys=["t1","t2", "pd"]),
        transforms.Orientationd(keys=["t1","t2", "pd"], axcodes="RAI", allow_missing_keys=True),
        transforms.Resized(keys=["t1","t2", "pd"], spatial_size=(128, 128, 128), allow_missing_keys=True),
        # transforms.CropForegroundd(keys=["t1","t2", "pd"], source_key="t1", allow_missing_keys=True),
        # transforms.SpatialPadd(keys=["t1","t2", "pd"], spatial_size=(128, 128, 128), allow_missing_keys=True),
        # transforms.RandSpatialCropd( keys=["t1","t2", "pd"],
        #     roi_size=(128, 128, 128),
        #     random_center=True, 
        #     random_size=False,
        # ),
        transforms.Lambdad(keys=["t1", "t2", "pd"], func=NormalizeZero()),
        transforms.ScaleIntensityRangePercentilesd(keys=["t1","t2", "pd"], lower=0, upper=99.75, b_min=0, b_max=1),
        transforms.ToTensord(keys=["t1","t2", "pd"]),
    ]
)

ixi_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["t1","t2", "pd"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["t1","t2", "pd"], allow_missing_keys=True),
        transforms.Lambdad(keys=["t1","t2", "pd"], func=lambda x: x[0, :, :, :]),
        transforms.EnsureChannelFirstd(keys=["t1","t2", "pd"], allow_missing_keys=True),
        transforms.EnsureTyped(keys=["t1","t2", "pd"]),
        transforms.Orientationd(keys=["t1","t2", "pd"], axcodes="RAI", allow_missing_keys=True),
        # IXI is skull-stripped: crop to the brain (with margin) first, then resize to 128^3.
        # transforms.CropForegroundd(keys=["t1","t2", "pd"], source_key="t1", margin=8, allow_missing_keys=True),
        transforms.Resized(keys=["t1","t2", "pd"], spatial_size=(128, 128, 128), allow_missing_keys=True),
        transforms.Lambdad(keys=["t1", "t2", "pd"], func=NormalizeZero()),
        transforms.ScaleIntensityRangePercentilesd(keys=["t1","t2", "pd"], lower=0, upper=99.75, b_min=0, b_max=1),
        transforms.ToTensord(keys=["t1","t2", "pd"]),
    ]
)

def infer_dataset_type(data_path):
    path_lower = data_path.lower()
    if "brats" in path_lower:
        return "brats"
    if "adni" in path_lower:
        return "adni"
    if "ixi" in path_lower:
        return "ixi"
    raise ValueError(
        f"Cannot infer dataset type from data_path={data_path!r}. "
        "Path should contain 'BraTS', 'ADNI' or 'IXI'."
    )


def get_split_for_phase(phase):
    phase = phase.lower()
    if phase in ("train", "training"):
        return "train"
    if phase in ("val", "valid", "validation"):
        return "val"
    if phase in ("test", "testing"):
        return "test"
    raise ValueError(
        f"Unknown phase={phase!r}. Expected one of train, val, test."
    )


def _resolve_json_path(data_path):
    if data_path.endswith(".json"):
        json_path = os.path.abspath(data_path)
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Dataset JSON not found: {json_path}")
        return json_path

    dataset_type = infer_dataset_type(data_path)
    json_name = {
        "brats": "BraTS2021_datalist_3D.json",
        "adni": "ADNI_datalist_3D.json",
        "ixi": "IXI_datalist_3D.json",
    }[dataset_type]
    json_path = os.path.join(data_path, json_name)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Dataset JSON not found: {json_path}")
    return json_path


def _get_split_base_dir(data_root, split, dataset_type):
    split_dir = DATASET_CONFIG[dataset_type]["split_3d_map"][split]
    base_dir = os.path.join(data_root, split_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Split directory not found: {base_dir}")
    return base_dir


def load_datalist_entries(json_path, split, dataset_type):
    data_root = os.path.dirname(json_path)
    json_key = SPLIT_TO_JSON_KEY[split]
    base_dir = _get_split_base_dir(data_root, split, dataset_type)
    entries = load_decathlon_datalist(
        json_path,
        is_segmentation=False,
        data_list_key=json_key,
        base_dir=base_dir,
    )
    print(
        f"loaded {len(entries)} samples from {json_path} "
        f"({json_key}, base_dir={base_dir})"
    )
    return entries


def get_dataset(data_path, phase="train"):
    json_path = _resolve_json_path(data_path)
    dataset_type = infer_dataset_type(json_path)
    split = get_split_for_phase(phase)
    transform = {
        "brats": brats_transforms,
        "adni": adni_transforms,
        "ixi": ixi_transforms,
    }[dataset_type]
    entries = load_datalist_entries(json_path, split, dataset_type)
    return MonaiDataset(data=entries, transform=transform)


class ModalityBase(Dataset):
    def __init__(self, data_path, phase="train", source=None, target=None, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.phase = phase
        self.source = source
        self.target = target
        json_path = _resolve_json_path(data_path)
        dataset_type = infer_dataset_type(json_path)
        self.modalities = DATASET_CONFIG[dataset_type]["keys"]
        self.data = get_dataset(data_path, phase=phase)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = dict(self.data[i])

        if self.source is None:
            source, target = np.random.choice(self.modalities, size=2, replace=False)
        else:
            source, target = self.source, self.target

        item["source"] = item[source]
        item["target"] = item[target]
        item["target_class"] = torch.tensor(self.modalities.index(target))
        return item


class CustomTrain(ModalityBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, phase="train", **kwargs)


class CustomVal(ModalityBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, phase="val", **kwargs)


class CustomTest(ModalityBase):
    def __init__(self, data_path, phase="test", **kwargs):
        super().__init__(data_path=data_path, phase=phase, **kwargs)


class ADNITrain(ModalityBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, phase="train", **kwargs)


class ADNIVal(ModalityBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, phase="val", **kwargs)


class ADNITest(ModalityBase):
    def __init__(self, data_path, phase="test", **kwargs):
        super().__init__(data_path=data_path, phase=phase, **kwargs)


class IXITrain(ModalityBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, phase="train", **kwargs)


class IXIVal(ModalityBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, phase="val", **kwargs)


class IXITest(ModalityBase):
    def __init__(self, data_path, phase="test", **kwargs):
        super().__init__(data_path=data_path, phase=phase, **kwargs)


class BraTSTrain(ModalityBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, phase="train", **kwargs)


class BraTSVal(ModalityBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, phase="val", **kwargs)


class BraTSTest(ModalityBase):
    def __init__(self, data_path, phase="test", **kwargs):
        super().__init__(data_path=data_path, phase=phase, **kwargs)