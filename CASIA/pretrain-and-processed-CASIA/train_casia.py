#!/usr/bin/env python3
#python train_casia.py --feat-path "C:\Users\admin\Desktop\DATA\fix_CASIA\processed_features_clean\train"

import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time
import json
from datetime import datetime

# 导入专为CASIA设计的加载器
from dataload_casia import create_casia_dataloaders
from model import BaseModel

# 复用train_for_clean.py中的辅助函数
from train_for_clean import EarlyStopper, print_gpu_usage, clear_gpu_memory, train_one_epoch, validate_one_epoch, test_fold_model, plot_training_history

def train_casia_loso_cv(config=None):
    """
    为CASIA数据集执行留一说话人交叉验证 (4-Fold)
    """
    if config is None:
        from config import TrainingConfig
        config = TrainingConfig
    
    print("=" * 80)
    print("CASIA Emotion Recognition - 4-Fold Leave-One-Speaker-Out CV")
    print("Using emotion2vec pretrained features")
    print(f"Max Epochs: {config.MAX_EPOCHS}, Early Stopping Patience: {config.EARLY_STOPPING_PATIENCE}")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() and config.USE_CUDA else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available() and config.USE_CUDA:
        print(f"GPU: {torch.cuda.get_device_name(0) if config.USE_CUDA else 'CPU'}")
    else:
        print("WARNING: CUDA not available or disabled. Aborting.")
        return
        
    all_fold_results = []
    
    # 创建总的保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_save_dir = os.path.join(config.SAVE_DIR, f"casia_loso_{timestamp}")
    os.makedirs(main_save_dir, exist_ok=True)
    print(f"Models and results will be saved in: {main_save_dir}")

    # 4折交叉验证
    for fold in range(4):
        fold_start_time = time.time()
        print(f"\n" + "="*80)
        print(f"--- Fold {fold+1}/4 ---")
        print("="*80)
        
        clear_gpu_memory()
        
        # 使用新的CASIA数据加载器
        train_loader, val_loader, test_loader, num_classes, idx2label = create_casia_dataloaders(
            config=config, fold=fold
        )
        
        model = BaseModel(input_dim=config.INPUT_DIM, output_dim=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss().to(device)
        
        # 配置学习率调度器 (与原脚本逻辑一致)
        if config.LR_SCHEDULER_TYPE == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.COSINE_T_0, T_mult=config.COSINE_T_MULT, eta_min=config.COSINE_ETA_MIN)
            scheduler_step_per_epoch = True
        else: # Default to ReduceLROnPlateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.LR_SCHEDULER_FACTOR, patience=config.LR_SCHEDULER_PATIENCE, min_lr=config.LR_SCHEDULER_MIN_LR)
            scheduler_step_per_epoch = False

        early_stopper = EarlyStopper(patience=config.EARLY_STOPPING_PATIENCE, min_delta=config.EARLY_STOPPING_MIN_DELTA, mode=config.EARLY_STOPPING_MODE)
        
        best_val_metric = float('-inf') if config.EARLY_STOPPING_MODE == 'max' else float('inf')
        best_model_state = None
        
        # 训练循环
        for epoch in range(config.MAX_EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_weighted_acc, val_f1 = validate_one_epoch(model, val_loader, criterion, device)
            
            # 更新学习率
            if scheduler_step_per_epoch:
                scheduler.step()
            else:
                scheduler.step(val_loss)

            print(f"Epoch {epoch+1:03d}/{config.MAX_EPOCHS} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, WA: {val_weighted_acc:.4f}, F1: {val_f1:.4f}")

            # 早停检查
            current_metric = locals()[config.EARLY_STOPPING_METRIC]
            if early_stopper(current_metric, epoch):
                print(f"Early stopping triggered at epoch {epoch+1}. Best score was {early_stopper.best_score:.4f} at epoch {early_stopper.best_epoch+1}.")
                break
            
            # 保存最佳模型
            is_better = (current_metric > best_val_metric) if config.EARLY_STOPPING_MODE == 'max' else (current_metric < best_val_metric)
            if is_better:
                best_val_metric = current_metric
                best_model_state = model.state_dict()
                print(f"🎉 New best model found at epoch {epoch+1} with {config.EARLY_STOPPING_METRIC}: {current_metric:.4f}")

        # 保存当前折的最佳模型
        fold_save_path = os.path.join(main_save_dir, f'best_model_fold_{fold+1}.ckpt')
        if best_model_state:
            torch.save(best_model_state, fold_save_path)
            print(f"💾 Best model for fold {fold+1} saved to {fold_save_path}")
        
        # 加载最佳模型进行测试
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # 在测试集上评估
        print(f"\n--- Testing on Fold {fold+1} ---")
        fold_acc, fold_weighted_acc, fold_f1, predictions, true_labels = test_fold_model(model, test_loader, device)
        
        # 生成分类报告
        from sklearn.metrics import classification_report
        fold_report = classification_report(true_labels, predictions, target_names=list(config.LABEL_DICT.keys()))
        print(fold_report)
        all_fold_results.append({
            'fold': fold + 1,
            'accuracy': fold_acc,
            'weighted_accuracy': fold_weighted_acc,
            'macro_f1': fold_f1,
            'report': fold_report,
        })
        
        fold_duration = time.time() - fold_start_time
        print(f"--- Fold {fold+1} finished in {fold_duration:.2f} seconds ---")

    # 汇总并打印最终结果
    print("\n" + "="*80)
    print("CASIA 4-Fold Cross-Validation Final Results")
    print("="*80)
    avg_acc = np.mean([res['accuracy'] for res in all_fold_results])
    avg_wa = np.mean([res['weighted_accuracy'] for res in all_fold_results])
    avg_f1 = np.mean([res['macro_f1'] for res in all_fold_results])
    
    print(f"Average Accuracy (UA): {avg_acc:.4f}")
    print(f"Average Weighted Accuracy (WA): {avg_wa:.4f}")
    print(f"Average Macro F1-Score: {avg_f1:.4f}")
    
    # 保存最终结果
    results_path = os.path.join(main_save_dir, 'final_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_fold_results, f, indent=4)
    print(f"\nDetailed results saved to {results_path}")

if __name__ == "__main__":
    from config import TrainingConfig, AdvancedConfig, CosineConfig, DebugConfig
    import argparse

    parser = argparse.ArgumentParser(description='CASIA Emotion Recognition Training (LOSO CV)')
    parser.add_argument('--config', type=str, default='default', choices=['default', 'advanced', 'cosine', 'debug'], help='Choose training configuration')
    parser.add_argument('--feat-path', type=str, required=True, help='Path to the preprocessed CASIA features file prefix (e.g., .../processed_features_clean/train)')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    
    args = parser.parse_args()

    # 选择配置
    if args.config == 'advanced':
        config = AdvancedConfig
    elif args.config == 'cosine':
        config = CosineConfig
    elif args.config == 'debug':
        config = DebugConfig
    else:
        config = TrainingConfig
    
    # 应用命令行参数覆盖
    config.FEAT_PATH = args.feat_path
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    
    config.print_config()

    train_casia_loso_cv(config) 