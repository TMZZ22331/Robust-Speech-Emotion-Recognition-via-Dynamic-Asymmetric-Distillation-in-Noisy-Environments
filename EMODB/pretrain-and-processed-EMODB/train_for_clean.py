#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time
import json
from datetime import datetime

from data import load_ssl_features, SpeechDataset
from model import BaseModel

# 修改为10折交叉验证，每个说话人单独一折
EMODB_SPEAKERS = ['03', '08', '09', '10', '11', '12', '13', '14', '15', '16']

def get_emodb_fold_speakers(fold_id):
    """
    获取指定fold的训练、验证、测试说话人
    10折交叉验证：8个说话人训练、1个说话人验证、1个说话人测试
    
    Args:
        fold_id (int): fold编号 (0-9)
    
    Returns:
        tuple: (train_speakers, val_speaker, test_speaker)
    """
    if fold_id < 0 or fold_id >= 10:
        raise ValueError(f"fold_id must be between 0 and 9, got {fold_id}")
    
    all_speakers = EMODB_SPEAKERS.copy()
    
    # 测试说话人：当前fold对应的说话人
    test_speaker = all_speakers[fold_id]
    
    # 验证说话人：下一个说话人（环形）
    val_speaker = all_speakers[(fold_id + 1) % 10]
    
    # 训练说话人：其余8个说话人
    train_speakers = [spk for spk in all_speakers if spk not in [test_speaker, val_speaker]]
    
    return train_speakers, val_speaker, test_speaker

def print_gpu_usage():
    """打印GPU使用情况"""
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        max_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        cached_memory = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"GPU Memory - Current: {current_memory:.2f}GB, Max: {max_memory:.2f}GB, Cached: {cached_memory:.2f}GB")

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

class EarlyStopper:
    """早停机制"""
    def __init__(self, patience=20, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # 'max' for accuracy, 'min' for loss
        self.counter = 0
        
        if mode == 'max':
            self.best_score = float('-inf')
            self.is_better = lambda score, best: score > best + self.min_delta
        else:  # mode == 'min'
            self.best_score = float('inf')
            self.is_better = lambda score, best: score < best - self.min_delta
            
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

def train_with_early_stopping(config=None):
    """
    使用早停机制训练emotion2vec分类模型
    """
    # 导入默认配置（如果没有提供配置）
    if config is None:
        from config import TrainingConfig
        config = TrainingConfig
    
    # 获取数据集名称（用于文件命名）
    dataset_name = getattr(config, 'DATASET_NAME', 'iemocap')
    
    print("=" * 80)
    print(f"{dataset_name.upper()} Emotion Recognition - Training with Early Stopping")
    print("Using emotion2vec pretrained features")
    print(f"Max Epochs: {config.MAX_EPOCHS}, Early Stopping Patience: {config.EARLY_STOPPING_PATIENCE}")
    print("=" * 80)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and config.USE_CUDA else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available() and config.USE_CUDA:
        print(f"GPU: {torch.cuda.get_device_name()}")
        clear_gpu_memory()
        print_gpu_usage()
        if config.CUDA_BENCHMARK:
            torch.backends.cudnn.benchmark = True
    else:
        print("WARNING: CUDA not available or disabled, using CPU!")
        if not config.USE_CUDA:
            print("(CUDA disabled in config)")
        return
    
    # 情绪标签映射
    label_dict = config.LABEL_DICT
    idx2label = {v: k for k, v in label_dict.items()}
    num_classes = len(label_dict)
    
    print(f"Emotion classes: {label_dict}")
    
    # 加载数据
    feat_path = config.FEAT_PATH
    print(f"Loading features from: {feat_path}")
    
    try:
        dataset = load_ssl_features(feat_path, label_dict)
        print(f"Loaded dataset with {dataset['num']} samples")
        
        # 统计类别分布
        labels = dataset['labels']
        class_counts = {idx2label[i]: labels.count(i) for i in range(num_classes)}
        print("\nClass distribution:")
        for emotion, count in class_counts.items():
            print(f"  {emotion}: {count} samples ({count/len(labels)*100:.1f}%)")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # 10折交叉验证配置 (EMODB) / 5折交叉验证配置 (其他)
    print("\n" + "=" * 60)
    is_emodb_speaker_isolation = (config.DATASET_NAME.lower() == 'emodb' and len(dataset.get('speakers', [])) > 0)
    if is_emodb_speaker_isolation:
        print("10-Fold Cross Validation with Early Stopping (EMODB)")
        n_folds = 10
    else:
        print("5-Fold Cross Validation with Early Stopping")
        n_folds = 5
    print("=" * 60)
    
    n_samples = config.SESSION_SAMPLES  # 其他数据集的session划分
    fold_results = []
    fold_weighted_results = []
    fold_f1_results = []
    all_predictions = []
    all_true_labels = []
    training_history = {}
    
    # 创建保存目录
    save_dir = config.SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    if is_emodb_speaker_isolation:
        print("\n" + "=" * 60)
        print("EMODB 10-Fold Cross Validation (Speaker-Isolated)")
        print("=" * 60)

    for fold in range(n_folds):
        if is_emodb_speaker_isolation:
             print(f"\n--- Fold {fold+1}/{n_folds} ---")
        else:
            print(f"\n--- Fold {fold+1}/{n_folds} (Test on Session {fold+1}) ---")
        
        # 清理GPU内存
        clear_gpu_memory()
        print_gpu_usage()
        
        # 根据数据集类型创建数据加载器
        if is_emodb_speaker_isolation:
            train_loader, val_loader, test_loader = create_emodb_speaker_isolated_loaders(
                dataset, fold, batch_size=config.BATCH_SIZE
            )
        else:
            # 原有的基于索引的划分逻辑（用于IEMOCAP等）
            test_start = sum(n_samples[:fold])
            test_end = test_start + n_samples[fold]
            print(f"Test samples: {test_start} to {test_end} (total: {n_samples[fold]})")
            train_loader, val_loader, test_loader = create_fold_loaders_with_validation(
                dataset, test_start, test_end, batch_size=config.BATCH_SIZE, 
                val_ratio=config.VALIDATION_RATIO, device=device
            )
        
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
        
        # 创建模型
        model = BaseModel(input_dim=config.INPUT_DIM, output_dim=num_classes).to(device)
        print(f"Model device: {next(model.parameters()).device}")
        
        # 训练配置
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss().to(device)
        
        # 根据配置选择学习率调度器
        if config.LR_SCHEDULER_TYPE == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=config.COSINE_T_0, T_mult=config.COSINE_T_MULT, 
                eta_min=config.COSINE_ETA_MIN, verbose=True
            )
            scheduler_step_per_epoch = True  # 每个epoch都step
        elif config.LR_SCHEDULER_TYPE == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=config.LR_SCHEDULER_PATIENCE, gamma=config.LR_SCHEDULER_FACTOR, verbose=True
            )
            scheduler_step_per_epoch = True
        else:  # ReduceLROnPlateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=config.LR_SCHEDULER_FACTOR, 
                patience=config.LR_SCHEDULER_PATIENCE, min_lr=config.LR_SCHEDULER_MIN_LR, verbose=True
            )
            scheduler_step_per_epoch = False  # 基于validation loss
        
        # 早停机制
        early_stopper = EarlyStopper(
            patience=config.EARLY_STOPPING_PATIENCE, 
            min_delta=config.EARLY_STOPPING_MIN_DELTA,
            mode=config.EARLY_STOPPING_MODE
        )
        
        # 训练历史记录
        fold_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_weighted_acc': [], 'val_f1': [],
            'epochs': [], 'lr': []
        }
        
        # 初始化最佳指标
        if config.EARLY_STOPPING_METRIC in ["val_acc", "val_weighted_acc", "val_f1"]:
            best_val_metric = float('-inf')
            is_better = lambda new, best: new > best
            if config.EARLY_STOPPING_METRIC == "val_acc":
                metric_name = "Val Acc"
            elif config.EARLY_STOPPING_METRIC == "val_weighted_acc":
                metric_name = "Val Weighted Acc"
            else:  # val_f1
                metric_name = "Val F1"
        else:  # val_loss
            best_val_metric = float('inf')
            is_better = lambda new, best: new < best
            metric_name = "Val Loss"
            
        best_model_state = None
        best_epoch = 0
        
        print(f"Starting training for fold {fold+1}...")
        print_gpu_usage()
        
        # 训练循环
        for epoch in range(config.MAX_EPOCHS):  # 最多MAX_EPOCHS轮
            epoch_start_time = time.time()
            
            # 训练阶段
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            
            # 验证阶段
            val_loss, val_acc, val_weighted_acc, val_f1 = validate_one_epoch(model, val_loader, criterion, device)
            
            # 学习率调度
            if scheduler_step_per_epoch:
                scheduler.step()  # 余弦退火等每个epoch都step
            else:
                scheduler.step(val_loss)  # ReduceLROnPlateau基于validation loss
            current_lr = optimizer.param_groups[0]['lr']
            
            # 保存历史
            fold_history['train_loss'].append(train_loss)
            fold_history['train_acc'].append(train_acc)
            fold_history['val_loss'].append(val_loss)
            fold_history['val_acc'].append(val_acc)
            fold_history['val_weighted_acc'].append(val_weighted_acc)
            fold_history['val_f1'].append(val_f1)
            fold_history['epochs'].append(epoch + 1)
            fold_history['lr'].append(current_lr)
            
            # 选择当前的验证指标
            if config.EARLY_STOPPING_METRIC == "val_acc":
                current_val_metric = val_acc
            elif config.EARLY_STOPPING_METRIC == "val_weighted_acc":
                current_val_metric = val_weighted_acc
            elif config.EARLY_STOPPING_METRIC == "val_f1":
                current_val_metric = val_f1
            else:  # val_loss
                current_val_metric = val_loss
            
            # 保存最佳模型
            if is_better(current_val_metric, best_val_metric):
                best_val_metric = current_val_metric
                best_model_state = model.state_dict().copy()
                best_epoch = epoch + 1
                
                # 保存最佳权重
                best_model_path = os.path.join(save_dir, f"best_model_fold_{fold+1}.ckpt")
                torch.save(best_model_state, best_model_path)
            
            epoch_time = time.time() - epoch_start_time
            
            # 根据配置显示进度
            if (epoch + 1) % config.PRINT_EVERY_N_EPOCHS == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{config.MAX_EPOCHS}: "
                      f"Train Loss={train_loss:.4f} Acc={train_acc:.2f}%")
                print(f"           Val Loss={val_loss:.4f} | Acc={val_acc:.2f}% | "
                      f"W-Acc={val_weighted_acc:.2f}% | F1={val_f1:.2f}% | LR={current_lr:.2e}")
                if (epoch + 1) % config.GPU_MONITOR_EVERY_N_EPOCHS == 0:
                    print_gpu_usage()
            
            # 早停检查
            if early_stopper(current_val_metric, epoch + 1):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                print(f"Best {metric_name}: {best_val_metric:.4f} at epoch {best_epoch}")
                break
        
        # 加载最佳模型进行测试
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 测试最终模型
        test_acc, test_weighted_acc, test_f1, fold_predictions, fold_true = test_fold_model(model, test_loader, device)
        fold_results.append(test_acc)
        fold_weighted_results.append(test_weighted_acc)
        fold_f1_results.append(test_f1)
        all_predictions.extend(fold_predictions)
        all_true_labels.extend(fold_true)
        
        # 保存训练历史
        training_history[f'fold_{fold+1}'] = fold_history
        
        print(f"\nFold {fold+1} Results:")
        print(f"  Best Epoch: {best_epoch}")
        print(f"  Best {metric_name}: {best_val_metric:.4f}")
        print(f"  Final Test Results:")
        print(f"    - Accuracy: {test_acc:.2f}%")
        print(f"    - Weighted Accuracy: {test_weighted_acc:.2f}%")
        print(f"    - F1 Score: {test_f1:.2f}%")
        print(f"  Total Epochs Trained: {len(fold_history['epochs'])}")
        print(f"  Scheduler: {config.LR_SCHEDULER_TYPE}")
        print(f"  Early Stop Metric: {config.EARLY_STOPPING_METRIC}")
        
        # 清理内存
        del model, optimizer, scheduler
        clear_gpu_memory()
    
    # 计算总体结果
    print("\n" + "=" * 60)
    print("Final Results Summary")
    print("=" * 60)
    
    # 计算各种指标的统计信息
    avg_accuracy = np.mean(fold_results)
    std_accuracy = np.std(fold_results)
    avg_weighted_accuracy = np.mean(fold_weighted_results)
    std_weighted_accuracy = np.std(fold_weighted_results)
    avg_f1 = np.mean(fold_f1_results)
    std_f1 = np.std(fold_f1_results)
    
    print(f"5-Fold Cross Validation Results:")
    print(f"  {'Fold':<6} {'Acc (%)':<8} {'W-Acc (%)':<10} {'F1 (%)':<8}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*8}")
    for i in range(5):
        print(f"  {i+1:<6} {fold_results[i]:<8.2f} {fold_weighted_results[i]:<10.2f} {fold_f1_results[i]:<8.2f}")
    
    print(f"\nOverall Performance:")
    print(f"  Accuracy:         {avg_accuracy:.2f}% ± {std_accuracy:.2f}%")
    print(f"  Weighted Accuracy: {avg_weighted_accuracy:.2f}% ± {std_weighted_accuracy:.2f}%")
    print(f"  F1 Score:         {avg_f1:.2f}% ± {std_f1:.2f}%")
    
    print(f"\nBest/Worst Performance:")
    print(f"  Best Fold (Acc):     Fold {np.argmax(fold_results)+1} ({max(fold_results):.2f}%)")
    print(f"  Worst Fold (Acc):    Fold {np.argmin(fold_results)+1} ({min(fold_results):.2f}%)")
    print(f"  Best Fold (W-Acc):   Fold {np.argmax(fold_weighted_results)+1} ({max(fold_weighted_results):.2f}%)")
    print(f"  Best Fold (F1):      Fold {np.argmax(fold_f1_results)+1} ({max(fold_f1_results):.2f}%)")
    
    # 详细分析
    print("\n" + "=" * 60)
    print("Detailed Classification Analysis")
    print("=" * 60)
    
    # 分类报告
    target_names = [idx2label[i] for i in range(num_classes)]
    report = classification_report(all_true_labels, all_predictions, 
                                   target_names=target_names, digits=4)
    print("Classification Report:")
    print(report)
    
    # 混淆矩阵
    cm = confusion_matrix(all_true_labels, all_predictions)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {dataset_name.upper()} Emotion Recognition\n(Early Stopping Training)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    confusion_matrix_path = os.path.join(save_dir, f'{dataset_name}_confusion_matrix.png')
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: {confusion_matrix_path}")
    
    # 每个类别的准确率
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    print(f"\nPer-class Accuracies:")
    for i, (emotion, acc) in enumerate(zip(target_names, class_accuracies)):
        print(f"  {emotion}: {acc*100:.2f}%")
    
    # 保存训练历史和结果
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'fold_accuracies': fold_results,
        'fold_weighted_accuracies': fold_weighted_results,
        'fold_f1_scores': fold_f1_results,
        'average_accuracy': float(avg_accuracy),
        'std_accuracy': float(std_accuracy),
        'average_weighted_accuracy': float(avg_weighted_accuracy),
        'std_weighted_accuracy': float(std_weighted_accuracy),
        'average_f1': float(avg_f1),
        'std_f1': float(std_f1),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_accuracies': {emotion: float(acc*100) for emotion, acc in zip(target_names, class_accuracies)},
        'training_history': training_history
    }
    
    results_path = os.path.join(save_dir, 'final_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"Results saved: {results_path}")
    
    # 绘制训练历史图
    plot_training_history(training_history, save_dir, dataset_name)
    
    print(f"\nAll models and results saved in: {save_dir}/")
    print("GPU Status after training:")
    print_gpu_usage()


def create_emodb_speaker_isolated_loaders(dataset, fold, batch_size):
    """
    为EmoDB创建严格按说话人隔离的数据加载器 (上游版本)
    采用10折交叉验证：8个说话人训练、1个说话人验证、1个说话人测试
    """
    if fold < 0 or fold >= 10:
        raise ValueError(f"fold必须在0-9范围内，当前: {fold}")

    # 获取当前fold的说话人分配
    train_speakers, val_speaker, test_speaker = get_emodb_fold_speakers(fold)

    print("📊 EMODB Speaker Isolation Strategy (10-Fold Cross-Validation):")
    print(f"  - Fold: {fold + 1}/10")
    print(f"  - 🏋️  Train Speakers: {train_speakers} (8 speakers)")
    print(f"  - 🧐 Validation Speaker: {val_speaker} (1 speaker)")
    print(f"  - 🧪 Test Speaker: {test_speaker} (1 speaker)")

    def extract_speaker_id(speaker_full_id):
        return speaker_full_id.split('_')[-1]

    dataset_speaker_ids = np.array([extract_speaker_id(spk) for spk in dataset['speakers']])
    
    train_indices = np.where(np.isin(dataset_speaker_ids, train_speakers))[0]
    val_indices = np.where(dataset_speaker_ids == val_speaker)[0]
    test_indices = np.where(dataset_speaker_ids == test_speaker)[0]

    np.random.shuffle(train_indices) # 只打乱训练集

    print(f"  - 📊 Train samples: {len(train_indices)}")
    print(f"  - 📊 Validation samples: {len(val_indices)}")
    print(f"  - 📊 Test samples: {len(test_indices)}")

    # 验证说话人隔离
    train_spk_set = set(np.unique(dataset_speaker_ids[train_indices]))
    val_spk_set = set(np.unique(dataset_speaker_ids[val_indices]))
    test_spk_set = set(np.unique(dataset_speaker_ids[test_indices]))

    assert len(train_spk_set & val_spk_set) == 0, "Leakage: Train and validation sets have overlapping speakers."
    assert len(train_spk_set & test_spk_set) == 0, "Leakage: Train and test sets have overlapping speakers."
    assert len(val_spk_set & test_spk_set) == 0, "Leakage: Validation and test sets have overlapping speakers."
    print("  - ✅ Speaker isolation verified successfully.")

    def create_subset(indices):
        if len(indices) == 0:
            return None
        
        sub_labels = [dataset['labels'][i] for i in indices]
        sub_sizes = dataset['sizes'][indices]
        sub_offsets = dataset['offsets'][indices]
        
        new_feats_list = [dataset['feats'][offset:offset+size] for offset, size in zip(sub_offsets, sub_sizes)]
        
        if not new_feats_list:
            return None

        new_feats = np.concatenate(new_feats_list, axis=0)
        new_sizes = np.array([feat.shape[0] for feat in new_feats_list])
        new_offsets = np.concatenate([np.array([0]), np.cumsum(new_sizes)[:-1]], dtype=np.int64)
        
        return SpeechDataset(new_feats, new_sizes, new_offsets, sub_labels)

    train_dataset = create_subset(train_indices)
    val_dataset = create_subset(val_indices)
    test_dataset = create_subset(test_indices)

    def create_loader(subset_dataset, shuffle):
        if subset_dataset is None:
            return [] # 返回一个空的iterable
        return DataLoader(
            subset_dataset,
            batch_size=batch_size,
            collate_fn=subset_dataset.collator,
            num_workers=0,
            pin_memory=False,
            shuffle=shuffle
        )

    train_loader = create_loader(train_dataset, shuffle=True)
    val_loader = create_loader(val_dataset, shuffle=False)
    test_loader = create_loader(test_dataset, shuffle=False)
    
    return train_loader, val_loader, test_loader

def create_fold_loaders_with_validation(dataset, test_start, test_end, batch_size=64, val_ratio=0.1, device='cuda'):
    """创建带验证集的数据加载器 (用于IEMOCAP等非说话人隔离场景)"""
    
    feats = dataset['feats']
    sizes = dataset['sizes']
    offsets = dataset['offsets']
    labels = dataset['labels']
    
    # 测试集
    test_sizes = sizes[test_start:test_end]
    test_offsets = offsets[test_start:test_end]
    test_labels = labels[test_start:test_end]
    
    test_offset_start = test_offsets[0]
    test_offset_end = test_offsets[-1] + test_sizes[-1]
    test_feats = feats[test_offset_start:test_offset_end, :]
    test_offsets = test_offsets - test_offset_start
    
    # 训练集（排除测试集）
    train_sizes = np.concatenate([sizes[:test_start], sizes[test_end:]])
    train_labels = labels[:test_start] + labels[test_end:]
    train_feats = np.concatenate([feats[:test_offset_start, :], feats[test_offset_end:, :]], axis=0)
    
    # 从训练集中分出验证集
    n_train = len(train_labels)
    n_val = int(n_train * val_ratio)
    
    # 随机划分训练和验证集
    indices = np.random.permutation(n_train)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    # 验证集
    val_sizes = train_sizes[val_indices]
    val_labels = [train_labels[i] for i in val_indices]
    val_offsets = np.concatenate([np.array([0]), np.cumsum(val_sizes)[:-1]], dtype=np.int64)
    
    val_feat_start = 0
    val_feats = []
    for idx in val_indices:
        start = sum(train_sizes[:idx])
        end = start + train_sizes[idx]
        val_feats.append(train_feats[start:end, :])
    val_feats = np.concatenate(val_feats, axis=0)
    
    # 重新构建训练集
    final_train_sizes = train_sizes[train_indices]
    final_train_labels = [train_labels[i] for i in train_indices]
    final_train_offsets = np.concatenate([np.array([0]), np.cumsum(final_train_sizes)[:-1]], dtype=np.int64)
    
    final_train_feats = []
    for idx in train_indices:
        start = sum(train_sizes[:idx])
        end = start + train_sizes[idx]
        final_train_feats.append(train_feats[start:end, :])
    final_train_feats = np.concatenate(final_train_feats, axis=0)
    
    # 创建数据集
    train_dataset = SpeechDataset(final_train_feats, final_train_sizes, final_train_offsets, final_train_labels)
    val_dataset = SpeechDataset(val_feats, val_sizes, val_offsets, val_labels)
    test_dataset = SpeechDataset(test_feats, test_sizes, test_offsets, test_labels)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             collate_fn=train_dataset.collator, 
                             num_workers=0, pin_memory=False, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           collate_fn=val_dataset.collator, 
                           num_workers=0, pin_memory=False, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            collate_fn=test_dataset.collator, 
                            num_workers=0, pin_memory=False, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(train_loader):
        feats = batch['net_input']['feats'].to(device, non_blocking=True)
        padding_mask = batch['net_input']['padding_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(feats, padding_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
        # 清理中间变量
        del feats, padding_mask, labels, outputs, loss
        
        if batch_idx % 20 == 0:
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * total_correct / total_samples
    return avg_loss, accuracy


def validate_one_epoch(model, val_loader, criterion, device):
    """验证一个epoch，返回loss、acc、weighted_acc、f1"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            feats = batch['net_input']['feats'].to(device, non_blocking=True)
            padding_mask = batch['net_input']['padding_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            outputs = model(feats, padding_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            # 收集预测和真实标签
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            del feats, padding_mask, labels, outputs, loss, predicted
    
    # 计算各种指标
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    weighted_accuracy = balanced_accuracy_score(all_labels, all_predictions) * 100
    f1 = f1_score(all_labels, all_predictions, average='weighted') * 100
    
    return avg_loss, accuracy, weighted_accuracy, f1


def test_fold_model(model, test_loader, device):
    """测试模型，返回acc、weighted_acc、f1和预测结果"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            feats = batch['net_input']['feats'].to(device, non_blocking=True)
            padding_mask = batch['net_input']['padding_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            outputs = model(feats, padding_mask)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            del feats, padding_mask, labels, outputs, predicted
    
    # 计算各种指标
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    weighted_accuracy = balanced_accuracy_score(all_labels, all_predictions) * 100
    f1 = f1_score(all_labels, all_predictions, average='weighted') * 100
    
    return accuracy, weighted_accuracy, f1, all_predictions, all_labels


def plot_training_history(training_history, save_dir, dataset_name='iemocap'):
    """绘制训练历史，包含加权准确率和F1分数"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    for fold_idx in range(5):
        fold_key = f'fold_{fold_idx+1}'
        if fold_key not in training_history:
            continue
            
        history = training_history[fold_key]
        color = colors[fold_idx]
        
        # Loss图
        axes[0, 0].plot(history['epochs'], history['train_loss'], 
                       color=color, linestyle='-', alpha=0.7, label=f'Fold {fold_idx+1}')
        axes[0, 1].plot(history['epochs'], history['val_loss'], 
                       color=color, linestyle='-', alpha=0.7, label=f'Fold {fold_idx+1}')
        
        # 普通准确率图
        axes[1, 0].plot(history['epochs'], history['train_acc'], 
                       color=color, linestyle='-', alpha=0.7, label=f'Fold {fold_idx+1}')
        axes[1, 1].plot(history['epochs'], history['val_acc'], 
                       color=color, linestyle='-', alpha=0.7, label=f'Fold {fold_idx+1}')
        
        # 加权准确率和F1分数图
        axes[2, 0].plot(history['epochs'], history['val_weighted_acc'], 
                       color=color, linestyle='-', alpha=0.7, label=f'Fold {fold_idx+1}')
        axes[2, 1].plot(history['epochs'], history['val_f1'], 
                       color=color, linestyle='-', alpha=0.7, label=f'Fold {fold_idx+1}')
    
    # 设置图表
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[2, 0].set_title('Validation Weighted Accuracy')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Weighted Accuracy (%)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].set_title('Validation F1 Score')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('F1 Score (%)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training History - 5-Fold Cross Validation with Early Stopping', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    history_plot_path = os.path.join(save_dir, f'{dataset_name}_training_history.png')
    plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved: {history_plot_path}")
    
    # 额外绘制学习率图
    fig_lr, ax_lr = plt.subplots(1, 1, figsize=(10, 6))
    
    for fold_idx in range(5):
        fold_key = f'fold_{fold_idx+1}'
        if fold_key not in training_history:
            continue
            
        history = training_history[fold_key]
        color = colors[fold_idx]
        
        ax_lr.plot(history['epochs'], history['lr'], 
                  color=color, linestyle='-', alpha=0.7, label=f'Fold {fold_idx+1}')
    
    ax_lr.set_title('Learning Rate Schedule')
    ax_lr.set_xlabel('Epoch')
    ax_lr.set_ylabel('Learning Rate')
    ax_lr.set_yscale('log')
    ax_lr.legend()
    ax_lr.grid(True, alpha=0.3)
    
    plt.tight_layout()
    lr_plot_path = os.path.join(save_dir, f'{dataset_name}_learning_rate.png')
    plt.savefig(lr_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Learning rate plot saved: {lr_plot_path}")


if __name__ == "__main__":
    # 导入默认配置
    from config import TrainingConfig
    
    # 设置随机种子
    torch.manual_seed(TrainingConfig.RANDOM_SEED)
    np.random.seed(TrainingConfig.RANDOM_SEED)
    
    train_with_early_stopping() 