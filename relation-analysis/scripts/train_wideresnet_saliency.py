"""
WideResNet Training Script for Saliency-Based Relation Classification
Supports multiple fusion strategies for saliency and cross-attention maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import argparse
from tqdm import tqdm
import logging
from datetime import datetime
import os


# ==================== Dataset ====================
class SaliencyRelationDataset(Dataset):
    """Dataset for loading saliency and cross-attention maps"""
    
    def __init__(self, file_paths, labels, fusion_mode='concat', transform=None):
        """
        Args:
            file_paths: List of paths to .pt files
            labels: List of class labels
            fusion_mode: How to fuse saliency and cross-attention maps
                - 'concat': Concatenate along channel dimension (3+3=6 channels)
                - 'saliency_only': Use only saliency maps (3 channels)
                - 'attention_only': Use only cross-attention maps (3 channels)
                - 'add': Element-wise addition (3 channels)
                - 'multiply': Element-wise multiplication (3 channels)
                - 'max': Element-wise maximum (3 channels)
                - 'weighted': Weighted sum with learnable weights (3 channels)
            transform: Optional transforms to apply
        """
        self.file_paths = file_paths
        self.labels = labels
        self.fusion_mode = fusion_mode
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load the .pt file
        data = torch.load(self.file_paths[idx])
        
        saliency = data['saliency_maps']  # Shape: [3, 32, 32]
        attention = data['cross_attention_maps']  # Shape: [3, 32, 32]
        label = self.labels[idx]
        
        # Apply fusion strategy
        if self.fusion_mode == 'concat':
            # Concatenate: [6, 32, 32]
            fused = torch.cat([saliency, attention], dim=0)
        elif self.fusion_mode == 'saliency_only':
            fused = saliency
        elif self.fusion_mode == 'attention_only':
            fused = attention
        elif self.fusion_mode == 'add':
            fused = saliency + attention
        elif self.fusion_mode == 'multiply':
            fused = saliency * attention
        elif self.fusion_mode == 'max':
            fused = torch.max(torch.stack([saliency, attention]), dim=0)[0]
        elif self.fusion_mode == 'weighted':
            # Default weighted sum (0.6 saliency, 0.4 attention)
            fused = 0.6 * saliency + 0.4 * attention
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")
        
        # Apply transforms if any
        if self.transform:
            fused = self.transform(fused)
        
        return fused, label


# ==================== WideResNet Model ====================
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, num_classes=24, widen_factor=10, dropRate=0.3, in_channels=3):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(in_channels, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


# ==================== Data Loading ====================
def load_dataset(data_dir, test_size=0.15, val_size=0.15, random_state=42):
    """
    Load all .pt files and create stratified train/val/test splits
    
    Args:
        data_dir: Path to dataset directory (e.g., saliency_datasets/early_layers)
        test_size: Proportion of data for test set
        val_size: Proportion of remaining data for validation set
        random_state: Random seed for reproducibility
    
    Returns:
        train_files, val_files, test_files, train_labels, val_labels, test_labels, class_names
    """
    data_dir = Path(data_dir)
    
    # Find all class directories
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('class_')])
    
    all_files = []
    all_labels = []
    class_names = []
    
    print(f"Found {len(class_dirs)} classes")
    
    for class_dir in class_dirs:
        # Extract class info
        class_id = int(class_dir.name.split('_')[1])
        class_name = '_'.join(class_dir.name.split('_')[2:])
        class_names.append(class_name)
        
        # Load all .pt files from this class
        pt_files = sorted(list(class_dir.glob('sample_*.pt')))
        
        print(f"Class {class_id} ({class_name}): {len(pt_files)} samples")
        
        all_files.extend(pt_files)
        all_labels.extend([class_id] * len(pt_files))
    
    all_files = np.array(all_files)
    all_labels = np.array(all_labels)
    
    # First split: separate test set
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        all_files, all_labels, test_size=test_size, stratify=all_labels, random_state=random_state
    )
    
    # Second split: separate validation set from remaining data
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels, test_size=val_size_adjusted, 
        stratify=train_val_labels, random_state=random_state
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_files)} samples")
    print(f"  Val:   {len(val_files)} samples")
    print(f"  Test:  {len(test_files)} samples")
    print(f"  Total: {len(all_files)} samples")
    
    return train_files, val_files, test_files, train_labels, val_labels, test_labels, class_names


# ==================== Training & Evaluation ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels


# ==================== Plotting ====================
def plot_training_curves(history, save_dir):
    """Plot and save training curves"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_dir / 'training_curves.png'}")


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted label',
           ylabel='True label',
           title='Confusion Matrix')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=6)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


# ==================== Main Training Loop ====================
def main(args):
    # Setup
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.fusion_mode}_{args.depth}x{args.widen_factor}_{timestamp}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Load dataset
    logger.info("Loading dataset...")
    train_files, val_files, test_files, train_labels, val_labels, test_labels, class_names = load_dataset(
        args.data_dir, test_size=args.test_size, val_size=args.val_size, random_state=args.seed
    )
    
    # Save class names
    with open(output_dir / 'class_names.json', 'w') as f:
        json.dump(class_names, f, indent=2)
    
    # Determine input channels based on fusion mode
    in_channels = 6 if args.fusion_mode == 'concat' else 3
    
    # Create datasets
    train_dataset = SaliencyRelationDataset(train_files, train_labels, fusion_mode=args.fusion_mode)
    val_dataset = SaliencyRelationDataset(val_files, val_labels, fusion_mode=args.fusion_mode)
    test_dataset = SaliencyRelationDataset(test_files, test_labels, fusion_mode=args.fusion_mode)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    
    # Create model
    logger.info(f"Creating WideResNet-{args.depth}-{args.widen_factor} with {in_channels} input channels...")
    model = WideResNet(
        depth=args.depth,
        num_classes=len(class_names),
        widen_factor=args.widen_factor,
        dropRate=args.dropout,
        in_channels=in_channels
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {num_params:,} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                               weight_decay=args.weight_decay, nesterov=True)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, output_dir / 'best_model.pth')
            logger.info(f"Saved best model with val_acc: {val_acc:.4f}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, output_dir / f'checkpoint_epoch_{epoch}.pth')
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, output_dir / 'final_model.pth')
    
    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, output_dir)
    
    # Evaluate on test set with best model
    logger.info("\nEvaluating on test set with best model...")
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels_actual = validate(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels_actual, test_preds)
    plot_confusion_matrix(cm, class_names, output_dir / 'confusion_matrix.png')
    
    # Classification report
    report = classification_report(test_labels_actual, test_preds, target_names=class_names)
    logger.info(f"\nClassification Report:\n{report}")
    
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    
    # Save final results
    results = {
        'fusion_mode': args.fusion_mode,
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'test_loss': float(test_loss),
        'num_params': num_params,
        'config': vars(args)
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nTraining complete! Results saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train WideResNet on Saliency Dataset')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='saliency_datasets/early_layers',
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='wideresnet_experiments',
                       help='Output directory for results')
    parser.add_argument('--test_size', type=float, default=0.15,
                       help='Proportion of data for test set')
    parser.add_argument('--val_size', type=float, default=0.15,
                       help='Proportion of data for validation set')
    
    # Fusion arguments
    parser.add_argument('--fusion_mode', type=str, default='concat',
                       choices=['concat', 'saliency_only', 'attention_only', 'add', 'multiply', 'max', 'weighted'],
                       help='Fusion strategy for saliency and attention maps')
    
    # Model arguments
    parser.add_argument('--depth', type=int, default=28,
                       help='WideResNet depth (28, 34, 40, etc.)')
    parser.add_argument('--widen_factor', type=int, default=10,
                       help='WideResNet widen factor')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    main(args)
