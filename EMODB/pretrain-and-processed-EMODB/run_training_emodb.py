#!/usr/bin/env python3

#用于EmoDB数据集的训练启动脚本

#这是 run_training.py 的一个适配版，专门用于处理由 emodb_preprocessing.ps1 脚本生成的特征。

#运行方式:
    # 1. 首先运行emodb_preprocessing.ps1来生成特征文件。
    #    例如，生成干净特征:
    #    .\emodb_preprocessing.ps1 -EMODB_ROOT "C:\path\to\EmoDB\EmoDB Dataset_wav_datasets" -OutputBasePath "C:\path\to\output"
    #    这会在 "C:\path\to\output\processed_features_clean" 中生成特征。

    # 2. 使用下面的命令启动训练，并通过 --feat-path 指定特征路径
    #python run_training_emodb.py --feat-path "C:\Users\admin\Desktop\DATA\processed_features_EMODB\processed_features_clean" 

    #    对于带噪数据，同样指定对应的特征路径:
    #    python run_training_emodb.py --feat-path "C:\path\to\output\processed_features_noisy_10db" --config advanced


import argparse
import sys
import os
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TrainingConfig, AdvancedConfig, CosineConfig, DebugConfig, EmoDBConfig
from train_for_clean import train_with_early_stopping

def main():
    parser = argparse.ArgumentParser(description='EmoDB Emotion Recognition Training')
    parser.add_argument('--config', type=str, default='default', 
                       choices=['default', 'advanced', 'cosine', 'debug'],
                       help='Choose training configuration')
    parser.add_argument('--feat-path', type=str, required=True,
                       help='Path to the preprocessed EmoDB features (e.g., .../processed_features_clean)')
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
        print("🚀 Using Advanced Configuration for EmoDB")
    elif args.config == 'cosine':
        config = CosineConfig
        print("🌊 Using Cosine Annealing Configuration for EmoDB")
    elif args.config == 'debug':
        config = DebugConfig
        print("🐛 Using Debug Configuration for EmoDB")
    else:
        config = EmoDBConfig
        print("⚙️  Using EmoDB Default Configuration")
    
    # 创建EmoDB专用的带时间戳保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    emodb_save_dir = os.path.join(config.SAVE_DIR, f"emodb_loso_{timestamp}")
    os.makedirs(emodb_save_dir, exist_ok=True)
    config.SAVE_DIR = emodb_save_dir
    print(f"📁 EmoDB results will be saved to: {emodb_save_dir}")
    
    # 应用命令行参数覆盖
    if args.feat_path:
        config.FEAT_PATH = args.feat_path
        print(f"Feature path set to: {args.feat_path}")
    
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
    print("Ready to start EmoDB training!")
    print("="*60)
    
    if args.config != 'debug':
        user_input = input("Press Enter to continue or 'q' to quit: ")
        if user_input.lower() == 'q':
            print("Training cancelled.")
            return
    
    # 启动训练
    try:
        train_with_early_stopping(config)
        print("\n🎉 EmoDB Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main() 