"""Dataset and dataloader helpers for the early-layers saliency maps.

Assumptions:
- `root` points to a directory that contains one subdirectory per class (folder name = label).
- Each class folder contains image files (png/jpg/etc.).
- Alternatively, pass `csv_path` with two columns: `filename,label` to index images.

This module provides:
- `EarlyLayersDataset` : a torch.utils.data.Dataset that loads images and labels
- `create_index_from_folders` : helper to index files when using folder-per-class layout
- `stratified_split_index` : splits index into stratified train/val/test
- `get_dataloaders` : builds PyTorch dataloaders for train/val/test with transforms

Example usage:
    from early_layers_dataset import get_dataloaders
    dls, mapping = get_dataloaders(root, batch_size=32)
    train_dl = dls['train']

"""

from typing import List, Tuple, Dict, Optional
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


def is_pt_file(fname: str):
    return fname.lower().endswith('.pt')


def create_index_from_folders(root: str) -> Tuple[List[Tuple[str,int]], Dict[int,str]]:
    """Walk `root` and build a list of (filepath, label_idx) tuples and an idx->label mapping.

    Looks for immediate subdirectories of `root` and treats each subdirectory name as a label.
    Files directly under `root` are ignored. Subdirectories are sorted to get deterministic label indices.
    """
    classes = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    if not classes:
        raise ValueError(f"No class subdirectories found in {root}. Provide csv_path or check layout.")
    label_to_idx = {c: i for i, c in enumerate(classes)}
    samples: List[Tuple[str,int]] = []
    for c in classes:
        folder = os.path.join(root, c)
        for fname in sorted(os.listdir(folder)):
            if is_pt_file(fname):
                samples.append((os.path.join(folder, fname), label_to_idx[c]))
    if not samples:
        raise ValueError(f"No .pt files found under {root} (searched classes: {classes}).")
    idx_to_label = {v:k for k,v in label_to_idx.items()}
    return samples, idx_to_label


def read_index_from_csv(root: str, csv_path: str) -> Tuple[List[Tuple[str,int]], Dict[int,str]]:
    """Read CSV with `filename,label` rows. Filenames are relative to `root` or absolute paths.
    Returns samples and idx->label mapping.
    """
    import csv
    samples: List[Tuple[str,int]] = []
    labels = []
    rows = []
    with open(csv_path, newline='') as fh:
        reader = csv.reader(fh)
        for r in reader:
            if not r: 
                continue
            if len(r) < 2:
                raise ValueError("CSV must have at least two columns: filename,label")
            rows.append((r[0].strip(), r[1].strip()))
            labels.append(r[1].strip())
    unique_labels = sorted(list(set(labels)))
    label_to_idx = {l:i for i,l in enumerate(unique_labels)}
    for fname, lab in rows:
        path = fname if os.path.isabs(fname) else os.path.join(root, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV-referenced file not found: {path}")
        samples.append((path, label_to_idx[lab]))
    idx_to_label = {v:k for k,v in label_to_idx.items()}
    return samples, idx_to_label


def stratified_split_index(samples: List[Tuple[str,int]], train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    """Return three lists of samples (train, val, test) stratified by label.

    Uses two-stage train_test_split for reproducibility.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    paths = [p for p, l in samples]
    labels = [l for p, l in samples]
    # First split out train vs rest
    rest_frac = val_frac + test_frac
    train_paths, rest_paths, train_labels, rest_labels = train_test_split(
        paths, labels, test_size=rest_frac, random_state=seed, stratify=labels)
    if rest_frac == 0:
        return list(zip(train_paths, train_labels)), [], []
    # Then split rest into val and test
    if val_frac == 0:
        val_paths, val_labels = [], []
        test_paths, test_labels = rest_paths, rest_labels
    else:
        relative_val_frac = val_frac / rest_frac
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            rest_paths, rest_labels, test_size=(1-relative_val_frac), random_state=seed, stratify=rest_labels)
    train = list(zip(train_paths, train_labels))
    val = list(zip(val_paths, val_labels))
    test = list(zip(test_paths, test_labels))
    return train, val, test


class EarlyLayersDataset(Dataset):
    """PyTorch Dataset for saliency maps stored in .pt files.

    samples: list of (path, int_label)
    transform: torchvision transforms (optional, applied after loading tensor)
    saliency_key: key in the .pt dict to extract (default 'saliency_maps')
    """
    def __init__(self, samples: List[Tuple[str,int]], transform=None, saliency_key='saliency_maps'):
        self.samples = samples
        self.transform = transform
        self.saliency_key = saliency_key

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            img = data[self.saliency_key]  # shape [3, 32, 32]
        else:
            img = data
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


def get_default_transforms(img_size: int = 32, train: bool = True):
    """Transforms for tensor inputs (already in [C, H, W] format).
    
    Since .pt files contain tensors [3, 32, 32], we skip ToTensor.
    If img_size != 32, we resize. Normalization uses ImageNet stats.
    """
    if train:
        ops = []
        if img_size != 32:
            ops.append(transforms.Resize((img_size, img_size)))
        ops.extend([
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transforms.Compose(ops)
    else:
        ops = []
        if img_size != 32:
            ops.append(transforms.Resize((img_size, img_size)))
        ops.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        return transforms.Compose(ops)


def get_dataloaders(root: str,
                    batch_size: int = 32,
                    img_size: int = 32,
                    num_workers: int = 4,
                    csv_path: Optional[str] = None,
                    seed: int = 42,
                    pin_memory: bool = True) -> Tuple[Dict[str, DataLoader], Dict[int,str]]:
    """Create stratified train/val/test dataloaders from `root` containing .pt files.

    Returns: (dataloaders_dict, idx_to_label)
    dataloaders_dict has keys 'train','val','test' (val/test might be empty lists if fractions are zero).
    """
    if csv_path:
        samples, idx_to_label = read_index_from_csv(root, csv_path)
    else:
        samples, idx_to_label = create_index_from_folders(root)

    # deterministic shuffle
    random.seed(seed)
    random.shuffle(samples)

    train_idx, val_idx, test_idx = stratified_split_index(samples, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=seed)

    train_ds = EarlyLayersDataset(train_idx, transform=get_default_transforms(img_size, train=True))
    val_ds = EarlyLayersDataset(val_idx, transform=get_default_transforms(img_size, train=False))
    test_ds = EarlyLayersDataset(test_idx, transform=get_default_transforms(img_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}, idx_to_label


if __name__ == '__main__':
    # Quick local smoke test for .pt files. Update `root` to your folder.
    root = os.path.join(os.path.dirname(__file__), '..', 'early_layers')
    root = os.path.abspath(root)
    print('Looking for dataset in', root)
    try:
        dls, mapping = get_dataloaders(root, batch_size=8, img_size=32, num_workers=0)
        print(f'Found {len(mapping)} classes:')
        for i, name in sorted(mapping.items())[:5]:
            print(f'  {i}: {name}')
        print('\nDataset sizes:')
        print(f"  Train: {len(dls['train'].dataset)}")
        print(f"  Val: {len(dls['val'].dataset)}")
        print(f"  Test: {len(dls['test'].dataset)}")
        print('\nLoading one batch from train...')
        for xb, yb in dls['train']:
            print('  Batch shapes:', xb.shape, yb.shape)
            print('  Tensor range: [{:.3f}, {:.3f}]'.format(xb.min().item(), xb.max().item()))
            print('  Labels:', yb.tolist())
            break
        print('\nSmoke test passed!')
    except Exception as e:
        import traceback
        print('Smoke test failed:', e)
        traceback.print_exc()
