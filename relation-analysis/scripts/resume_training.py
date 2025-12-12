"""
Resume training from a checkpoint
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import json
from train_wideresnet_saliency import (
    load_dataset,
    SaliencyRelationDataset,
    WideResNet,
    train_epoch,
    validate,
    plot_training_curves
)
import logging
from tqdm import tqdm


def resume_training(args):
    """Resume training from a checkpoint"""
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    # Get experiment directory
    exp_dir = checkpoint_path.parent
    
    # Load class names and config
    with open(exp_dir / 'class_names.json', 'r') as f:
        class_names = json.load(f)
    
    with open(exp_dir / 'results.json', 'r') as f:
        config = json.load(f)['config']
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(exp_dir / 'training_resumed.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Load dataset
    logger.info("Loading dataset...")
    train_files, val_files, test_files, train_labels, val_labels, test_labels, _ = load_dataset(
        args.data_dir or config['data_dir'],
        test_size=config.get('test_size', 0.15),
        val_size=config.get('val_size', 0.15),
        random_state=config.get('seed', 42)
    )
    
    # Determine input channels
    fusion_mode = config['fusion_mode']
    in_channels = 6 if fusion_mode == 'concat' else 3
    
    # Create datasets
    train_dataset = SaliencyRelationDataset(train_files, train_labels, fusion_mode=fusion_mode)
    val_dataset = SaliencyRelationDataset(val_files, val_labels, fusion_mode=fusion_mode)
    test_dataset = SaliencyRelationDataset(test_files, test_labels, fusion_mode=fusion_mode)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size or config['batch_size'],
                             shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size or config['batch_size'],
                           shuffle=False, num_workers=config['num_workers'], pin_memory=True)
    
    # Create model
    logger.info(f"Creating model...")
    model = WideResNet(
        depth=config['depth'],
        num_classes=len(class_names),
        widen_factor=config['widen_factor'],
        dropRate=config['dropout'],
        in_channels=in_channels
    ).to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr or config['lr'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],
        nesterov=True
    )
    
    # Load optimizer state if continuing from same learning rate
    if not args.lr:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Create scheduler
    start_epoch = checkpoint['epoch']
    total_epochs = args.epochs or config['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - start_epoch
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Load existing history
    with open(exp_dir / 'history.json', 'r') as f:
        history = json.load(f)
    
    best_val_acc = checkpoint['val_acc']
    
    logger.info(f"Resuming from epoch {start_epoch}/{total_epochs}")
    logger.info(f"Previous best val acc: {best_val_acc:.4f}")
    
    # Continue training
    for epoch in range(start_epoch + 1, total_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{total_epochs}")
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
            }, exp_dir / 'best_model.pth')
            logger.info(f"Saved best model with val_acc: {val_acc:.4f}")
        
        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, exp_dir / f'checkpoint_epoch_{epoch}.pth')
    
    # Save updated history
    with open(exp_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, exp_dir)
    
    logger.info(f"\nTraining resumed and completed! Results saved to {exp_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resume WideResNet training from checkpoint')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (e.g., best_model.pth)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to dataset (if different from original)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Total epochs to train (default: use original config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (default: use original config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (default: use original config and optimizer state)')
    
    args = parser.parse_args()
    resume_training(args)
