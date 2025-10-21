#!/usr/bin/env python3
"""
简单的训练启动脚本
运行方式：
    python run_training.py              # 使用默认配置
    python run_training.py --advanced   # 使用高级配置
    python run_training.py --debug      # 使用调试配置
"""

import argparse
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TrainingConfig, AdvancedConfig, CosineConfig, DebugConfig
from train_for_clean import train_with_early_stopping

def main():
    parser = argparse.ArgumentParser(description='IEMOCAP Emotion Recognition Training')
    parser.add_argument('--config', type=str, default='default', 
                       choices=['default', 'advanced', 'cosine', 'debug'],
                       help='Choose training configuration')
    parser.add_argument('--feat-path', type=str, default=None,
                       help='Override feature path')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override max epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--patience', type=int, default=None,
                       help='Override early stopping patience')
    
    args = parser.parse_args()
    
    # 选择配置
    if args.config == 'advanced':
        config = AdvancedConfig
        print("🚀 Using Advanced Configuration")
    elif args.config == 'cosine':
        config = CosineConfig
        print("🌊 Using Cosine Annealing Configuration")
    elif args.config == 'debug':
        config = DebugConfig
        print("🐛 Using Debug Configuration")
    else:
        config = TrainingConfig
        print("⚙️  Using Default Configuration")
    
    # 应用命令行参数覆盖
    if args.feat_path:
        config.FEAT_PATH = args.feat_path
        print(f"Feature path overridden: {args.feat_path}")
    
    if args.epochs:
        config.MAX_EPOCHS = args.epochs
        print(f"Max epochs overridden: {args.epochs}")
    
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
        print(f"Batch size overridden: {args.batch_size}")
    
    if args.lr:
        config.LEARNING_RATE = args.lr
        print(f"Learning rate overridden: {args.lr}")
    
    if args.patience:
        config.EARLY_STOPPING_PATIENCE = args.patience
        print(f"Early stopping patience overridden: {args.patience}")
    
    # 显示配置信息
    config.print_config()
    
    # 确认开始训练
    print("\n" + "="*60)
    print("Ready to start training!")
    print("="*60)
    
    if args.config != 'debug':
        user_input = input("Press Enter to continue or 'q' to quit: ")
        if user_input.lower() == 'q':
            print("Training cancelled.")
            return
    
    # 启动训练
    try:
        train_with_early_stopping(config)
        print("\n🎉 Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main() 