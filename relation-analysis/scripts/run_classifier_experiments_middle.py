#!/usr/bin/env python
import json
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

ROOT = Path('..').resolve()
DATA_ROOT = ROOT.parent / 'saliency_datasets' / 'middle_layers'
assert DATA_ROOT.exists(), DATA_ROOT

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _normalize(x, strategy):
    if strategy == 'none':
        return x
    if strategy == 'minmax':
        xmin = x.amin(dim=(1, 2), keepdim=True)
        xmax = x.amax(dim=(1, 2), keepdim=True)
        return (x - xmin) / torch.clamp(xmax - xmin, min=1e-6)
    if strategy == 'zscore':
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        return (x - mean) / std
    raise ValueError(strategy)

def _smooth(x, kernel):
    if kernel <= 1:
        return x
    return F.avg_pool2d(x.unsqueeze(0), kernel_size=kernel, stride=1, padding=kernel // 2).squeeze(0)

def _apply_ablation(x, zero_subject=False, zero_predicate=False, zero_object=False, shuffle=False):
    # Channels: 0=subject, 1=predicate token, 2=object (as produced in data)
    if zero_subject and x.shape[0] >= 1:
        x[0] = 0
    if zero_predicate and x.shape[0] >= 2:
        x[1] = 0
    if zero_object and x.shape[0] >= 3:
        x[2] = 0
    if shuffle:
        perm = torch.randperm(x.shape[0])
        x = x[perm]
    return x

def fuse_inputs(sal, cross, mode):
    if mode == 'saliency':
        return sal
    if mode == 'cross':
        return cross
    if mode == 'concat':
        return torch.cat([sal, cross], dim=0)
    if mode == 'triple':
        return torch.cat([sal, cross, sal * cross], dim=0)
    if mode == 'diff':
        return torch.cat([sal, cross - sal], dim=0)
    if mode == 'late':
        return (sal, cross)
    raise ValueError(mode)

class SaliencyDataset(Dataset):
    def __init__(self, root: Path, mode: str, norm: str='none', smooth_kernel=0,
                 zero_subject=False, zero_predicate=False, zero_object=False, shuffle=False):
        self.root = Path(root)
        self.mode = mode
        self.norm = norm
        self.smooth_kernel = smooth_kernel
        self.zero_subject = zero_subject
        self.zero_predicate = zero_predicate
        self.zero_object = zero_object
        self.shuffle = shuffle
        self.files = sorted(self.root.glob('class_*/sample_*.pt'))
        if len(self.files) == 0:
            raise RuntimeError(f'No samples found under {self.root}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx])
        sal = sample['saliency_maps'].float()
        cross = sample['cross_attention_maps'].float()
        sal = _normalize(sal, self.norm)
        cross = _normalize(cross, self.norm)
        if self.smooth_kernel > 1:
            sal = _smooth(sal, self.smooth_kernel)
            cross = _smooth(cross, self.smooth_kernel)
        sal = _apply_ablation(sal, self.zero_subject, self.zero_predicate, self.zero_object, self.shuffle)
        cross = _apply_ablation(cross, self.zero_subject, self.zero_predicate, self.zero_object, self.shuffle)
        x = fuse_inputs(sal, cross, self.mode)
        y = int(sample['class_id'])
        return x, y

    def class_counts(self):
        return dict(sorted(Counter(int(torch.load(f)['class_id']) for f in self.files).items()))

def make_sampler(counts, files):
    total = sum(counts.values())
    class_weights = {c: total / (len(counts) * n) for c, n in counts.items()}
    weights = [class_weights[int(torch.load(f)['class_id'])] for f in files]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def build_loaders(root, mode, norm='none', smooth_kernel=0, batch_size=128, val_split=0.1, seed=42, balanced=False,
                 zero_subject=False, zero_predicate=False, zero_object=False, shuffle=False):
    ds = SaliencyDataset(root, mode=mode, norm=norm, smooth_kernel=smooth_kernel,
                         zero_subject=zero_subject, zero_predicate=zero_predicate,
                         zero_object=zero_object, shuffle=shuffle)
    n_total = len(ds)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)
    kwargs = dict(num_workers=2, pin_memory=True)
    if balanced:
        # compute weights only over train indices
        train_files = [ds.files[i] for i in train_ds.indices]
        sampler = make_sampler(ds.class_counts(), train_files)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = None if self.equalInOut else nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(x)))
        else:
            out = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(out)))
        if self.droprate>0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        shortcut = x if self.equalInOut else self.convShortcut(x)
        return torch.add(shortcut, out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super().__init__()
        layers=[block(i==0 and in_planes or out_planes, out_planes, i==0 and stride or 1, dropRate) for i in range(int(nb_layers))]
        self.layer=nn.Sequential(*layers)
    def forward(self,x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=8, dropRate=0.3, in_channels=3):
        super().__init__()
        nC=[16,16*widen_factor,32*widen_factor,64*widen_factor]
        n=(depth-4)/6
        block=BasicBlock
        self.conv1=nn.Conv2d(in_channels,nC[0],3,padding=1,bias=False)
        self.block1=NetworkBlock(n,nC[0],nC[1],block,1,dropRate)
        self.block2=NetworkBlock(n,nC[1],nC[2],block,2,dropRate)
        self.block3=NetworkBlock(n,nC[2],nC[3],block,2,dropRate)
        self.bn1=nn.BatchNorm2d(nC[3])
        self.relu=nn.ReLU(inplace=True)
        self.fc=nn.Linear(nC[3],num_classes)
        self.nChannels=nC[3]
    def forward(self,x):
        out=self.conv1(x)
        out=self.block1(out)
        out=self.block2(out)
        out=self.block3(out)
        out=self.relu(self.bn1(out))
        out=F.avg_pool2d(out,8)
        out=out.view(-1,self.nChannels)
        return self.fc(out)

class DualWRN(nn.Module):
    def __init__(self, depth=28, num_classes=24, widen_factor=8, dropRate=0.3):
        super().__init__()
        self.sal=WideResNet(depth,num_classes,widen_factor,dropRate,in_channels=3)
        self.cross=WideResNet(depth,num_classes,widen_factor,dropRate,in_channels=3)
    def forward(self,x_tuple):
        s,c=x_tuple
        return 0.5*(self.sal(s)+self.cross(c))


class TinyCNN(nn.Module):
    """Lightweight convnet for 32x32 inputs."""
    def __init__(self, in_channels, num_classes=24):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class SimpleMLP(nn.Module):
    """Flattened MLP baseline."""
    def __init__(self, in_channels, num_classes=24, hidden1=512, hidden2=256, dropout=0.3):
        super().__init__()
        input_dim = in_channels * 32 * 32
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def maybe_augment(x, hflip=False):
    if not hflip:
        return x
    if isinstance(x, (tuple, list)):
        return tuple(torch.flip(t, dims=[2]) for t in x)
    return torch.flip(x, dims=[2])


def move_to_device(x):
    if isinstance(x, (tuple, list)):
        return tuple(t.to(DEVICE) for t in x)
    return x.to(DEVICE)


def train_one_epoch(model, loader, optimizer, criterion, hflip=False):
    model.train()
    loss_sum=0; correct=0; total=0
    for xb,yb in tqdm(loader, leave=False):
        xb = maybe_augment(xb, hflip=hflip)
        xb = move_to_device(xb)
        yb = yb.to(DEVICE)
        optimizer.zero_grad()
        logits=model(xb)
        loss=criterion(logits,yb)
        loss.backward(); optimizer.step()
        loss_sum += loss.item()*yb.size(0)
        correct += (logits.argmax(1)==yb).sum().item()
        total += yb.size(0)
    return loss_sum/total, correct/total

def evaluate(model, loader, criterion, num_classes=24):
    model.eval()
    loss_sum=0; correct=0; total=0
    conf=torch.zeros(num_classes,num_classes,dtype=torch.long)
    with torch.no_grad():
        for xb,yb in loader:
            xb = move_to_device(xb)
            yb = yb.to(DEVICE)
            logits=model(xb)
            loss=criterion(logits,yb)
            loss_sum += loss.item()*yb.size(0)
            preds=logits.argmax(1)
            correct += (preds==yb).sum().item()
            total += yb.size(0)
            for t,p in zip(yb.view(-1), preds.view(-1)):
                conf[t.long(), p.long()] += 1
    per_class_acc = conf.diag() / conf.sum(1).clamp(min=1)
    return loss_sum/total, correct/total, per_class_acc.tolist(), conf.cpu()


def run_experiment(cfg):
    mode = cfg['mode']
    norm = cfg.get('norm','none')
    smooth = cfg.get('smooth',0)
    batch_size = cfg.get('batch_size',128)
    epochs = cfg.get('epochs',5)
    balanced = cfg.get('balanced',False)
    widen = cfg.get('widen_factor',8)
    drop = cfg.get('dropout',0.3)
    lr = cfg.get('lr',1e-3)
    wd = cfg.get('weight_decay',1e-4)
    hflip = cfg.get('hflip', False)
    zero_sub = cfg.get('zero_subject', False)
    zero_pred = cfg.get('zero_predicate', False)
    zero_obj = cfg.get('zero_object', False)
    shuffle = cfg.get('shuffle', False)
    save_ckpt = cfg.get('save_ckpt', True)
    arch = cfg.get('arch', 'wrn')

    train_loader, val_loader = build_loaders(
        DATA_ROOT, mode, norm=norm, smooth_kernel=smooth, batch_size=batch_size,
        balanced=balanced, zero_subject=zero_sub, zero_predicate=zero_pred,
        zero_object=zero_obj, shuffle=shuffle
    )
    in_ch_map={'saliency':3,'cross':3,'concat':6,'triple':9,'diff':6}
    if arch == 'wrn':
        if mode=='late':
            model=DualWRN(widen_factor=widen, dropRate=drop).to(DEVICE)
        else:
            in_ch=in_ch_map[mode]
            model=WideResNet(depth=28,num_classes=24,widen_factor=widen,dropRate=drop,in_channels=in_ch).to(DEVICE)
    elif arch == 'tiny_cnn':
        if mode=='late':
            model=DualWRN(widen_factor=widen, dropRate=drop).to(DEVICE)
        else:
            in_ch=in_ch_map[mode]
            model=TinyCNN(in_channels=in_ch, num_classes=24).to(DEVICE)
    elif arch == 'mlp':
        if mode=='late':
            model=DualWRN(widen_factor=widen, dropRate=drop).to(DEVICE)
        else:
            in_ch=in_ch_map[mode]
            model=SimpleMLP(in_channels=in_ch, num_classes=24, dropout=drop).to(DEVICE)
    else:
        raise ValueError(f'Unknown arch: {arch}')
    criterion=nn.CrossEntropyLoss()
    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched=CosineAnnealingLR(opt, T_max=epochs)

    history=[]
    for ep in range(1, epochs+1):
        tr_loss,tr_acc=train_one_epoch(model, train_loader, opt, criterion, hflip=hflip)
        val_loss,val_acc, per_class, conf = evaluate(model, val_loader, criterion)
        sched.step()
        history.append({'epoch':ep,'train_loss':tr_loss,'train_acc':tr_acc,'val_loss':val_loss,'val_acc':val_acc})
        print(f"{mode} | epoch {ep}/{epochs} | train_acc {tr_acc:.3f} val_acc {val_acc:.3f}")
    # top confusions
    mis = conf - torch.diag(torch.diag(conf))
    vals, idxs = torch.topk(mis.view(-1), k=10)
    top_confusions = []
    for v, idx in zip(vals, idxs):
        if v <= 0:
            continue
        true = int(idx // 24)
        pred = int(idx % 24)
        top_confusions.append({'true': true, 'pred': pred, 'count': int(v)})

    # save checkpoint
    if save_ckpt:
        ckpt_dir = Path('runs')
        ckpt_dir.mkdir(exist_ok=True)
        ablation_tag = ''.join([
            'S' if zero_sub else '',
            'P' if zero_pred else '',
            'O' if zero_obj else '',
            'R' if shuffle else ''
        ]) or 'none'
        ckpt_name = f"{arch}_mode-{mode}_norm-{norm}_smooth-{smooth}_abl-{ablation_tag}.pt"
        torch.save(model.state_dict(), ckpt_dir / ckpt_name)

    return {
        'cfg':cfg,
        'history':history,
        'final_val_acc':history[-1]['val_acc'],
        'per_class_acc':per_class,
        'top_confusions': top_confusions
    }


def main():
    base_modes = [
        {'name':'saliency','mode':'saliency'},
        {'name':'cross','mode':'cross'},
        {'name':'concat','mode':'concat'},
        {'name':'triple','mode':'triple'},
        {'name':'diff','mode':'diff'},
    ]
    configs=[
        # WRN (all modes including late)
        *[{**m, 'arch':'wrn'} for m in base_modes],
        {'name':'late','mode':'late','arch':'wrn'},
        # Tiny CNN baselines (same modes as base)
        *[{**m, 'name':f"{m['name']}_tinycnn", 'arch':'tiny_cnn'} for m in base_modes],
        # MLP baselines (same modes as base)
        *[{**m, 'name':f"{m['name']}_mlp", 'arch':'mlp'} for m in base_modes],
    ]
    results=[]
    for cfg in configs:
        print(f"\n=== Running {cfg['name']} ===")
        res=run_experiment({**cfg,
                            'epochs':100,
                            'batch_size':128,
                            'widen_factor':8,
                            'dropout':0.3,
                            'norm':'none',
                            'smooth':0,
                            'balanced':False,
                            'hflip':False,
                            'zero_subject':False,
                            'zero_predicate':False,
                            'zero_object':False,
                            'shuffle':False,
                            'save_ckpt':True})
        results.append(res)
    out_path=Path('runs')/ 'experiment_results.json'
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print('Saved results to', out_path.resolve())

if __name__=='__main__':
    main()
