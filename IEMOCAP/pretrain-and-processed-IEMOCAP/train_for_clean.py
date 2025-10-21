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

from data import load_ssl_features, get_cv_dataloaders
from model import BaseModel

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
    采用5折交叉验证：每个fold包含3个训练会话、1个验证会话、1个测试会话
    """
    if config is None:
        from config import TrainingConfig
        config = TrainingConfig
    
    dataset_name = getattr(config, 'DATASET_NAME', 'iemocap')
    
    print("=" * 80)
    print(f"{dataset_name.upper()} Emotion Recognition - 5-Fold Cross-Validation Training")
    print("Using emotion2vec pretrained features")
    print(f"Max Epochs: {config.MAX_EPOCHS}, Early Stopping Patience: {config.EARLY_STOPPING_PATIENCE}")
    print("Data Split: 3 Sessions (Train) + 1 Session (Val) + 1 Session (Test)")
    print("=" * 80)
    
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
    
    label_dict = config.LABEL_DICT
    idx2label = {v: k for k, v in label_dict.items()}
    num_classes = len(label_dict)
    
    print(f"Emotion classes: {label_dict}")
    
    feat_path = config.FEAT_PATH
    print(f"Loading features from: {feat_path}")
    
    try:
        dataset = load_ssl_features(feat_path, label_dict)
        print(f"Loaded dataset with {dataset['num']} samples")
        
        labels = dataset['labels']
        class_counts = {idx2label[i]: labels.count(i) for i in range(num_classes)}
        print("\nClass distribution:")
        for emotion, count in class_counts.items():
            print(f"  {emotion}: {count} samples ({count/len(labels)*100:.1f}%)")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print("\n" + "=" * 80)
    print("5-Fold Cross-Validation Training")
    print("=" * 80)
    
    fold_results = []
    fold_weighted_results = []
    fold_f1_results = []
    all_test_predictions = []
    all_test_true_labels = []
    training_history = {}
    
    save_dir = config.SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    for fold in range(5):
        fold_id = fold + 1
        print(f"\n{'='*60}")
        print(f"FOLD {fold_id}/5")
        print(f"{'='*60}")
        
        clear_gpu_memory()
        print_gpu_usage()
        
        # 获取三个数据加载器：训练、验证、测试
        train_loader, val_loader, test_loader = get_cv_dataloaders(
            dataset, batch_size=config.BATCH_SIZE, fold_id=fold_id
        )
        
        print(f"📊 Data Statistics:")
        print(f"   🏋️  Training batches: {len(train_loader)}")
        print(f"   🧐 Validation batches: {len(val_loader)}")
        print(f"   🧪 Test batches: {len(test_loader)}")
        
        model = BaseModel(input_dim=config.INPUT_DIM, output_dim=num_classes).to(device)
        print(f"Model device: {next(model.parameters()).device}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss().to(device)
        
        if config.LR_SCHEDULER_TYPE == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=config.COSINE_T_0, T_mult=config.COSINE_T_MULT, 
                eta_min=config.COSINE_ETA_MIN, verbose=True
            )
            scheduler_step_per_epoch = True
        elif config.LR_SCHEDULER_TYPE == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=config.LR_SCHEDULER_PATIENCE, gamma=config.LR_SCHEDULER_FACTOR, verbose=True
            )
            scheduler_step_per_epoch = True
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=config.LR_SCHEDULER_FACTOR, 
                patience=config.LR_SCHEDULER_PATIENCE, min_lr=config.LR_SCHEDULER_MIN_LR, verbose=True
            )
            scheduler_step_per_epoch = False
        
        early_stopper = EarlyStopper(
            patience=config.EARLY_STOPPING_PATIENCE, 
            min_delta=config.EARLY_STOPPING_MIN_DELTA,
            mode=config.EARLY_STOPPING_MODE
        )
        
        fold_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_weighted_acc': [], 'val_f1': [],
            'epochs': [], 'lr': []
        }
        
        if config.EARLY_STOPPING_METRIC in ["val_acc", "val_weighted_acc", "val_f1"]:
            best_val_metric = float('-inf')
            is_better = lambda new, best: new > best
            if config.EARLY_STOPPING_METRIC == "val_acc":
                metric_name = "Val Acc"
            elif config.EARLY_STOPPING_METRIC == "val_weighted_acc":
                metric_name = "Val Weighted Acc"
            else:
                metric_name = "Val F1"
        else:
            best_val_metric = float('inf')
            is_better = lambda new, best: new < best
            metric_name = "Val Loss"
            
        best_model_state = None
        best_epoch = 0
        
        print(f"\n🚀 Starting training for fold {fold_id}...")
        print_gpu_usage()
        
        for epoch in range(config.MAX_EPOCHS):
            epoch_start_time = time.time()
            
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_weighted_acc, val_f1 = validate_one_epoch(model, val_loader, criterion, device)
            
            if scheduler_step_per_epoch:
                scheduler.step()
            else:
                scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            fold_history['train_loss'].append(train_loss)
            fold_history['train_acc'].append(train_acc)
            fold_history['val_loss'].append(val_loss)
            fold_history['val_acc'].append(val_acc)
            fold_history['val_weighted_acc'].append(val_weighted_acc)
            fold_history['val_f1'].append(val_f1)
            fold_history['epochs'].append(epoch + 1)
            fold_history['lr'].append(current_lr)
            
            if config.EARLY_STOPPING_METRIC == "val_acc":
                current_val_metric = val_acc
            elif config.EARLY_STOPPING_METRIC == "val_weighted_acc":
                current_val_metric = val_weighted_acc
            elif config.EARLY_STOPPING_METRIC == "val_f1":
                current_val_metric = val_f1
            else:
                current_val_metric = val_loss
            
            if is_better(current_val_metric, best_val_metric):
                best_val_metric = current_val_metric
                best_epoch = epoch + 1
                best_model_state = model.state_dict().copy()
            
            print(f"Epoch {epoch+1:3d}/{config.MAX_EPOCHS} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"{metric_name}: {current_val_metric:.4f} (Best: {best_val_metric:.4f} @ epoch {best_epoch}) | "
                  f"LR: {current_lr:.6f} | Time: {time.time() - epoch_start_time:.1f}s")
            
            if early_stopper(current_val_metric, epoch + 1):
                print(f"⏹️  Early stopping at epoch {epoch + 1}. Best score: {best_val_metric:.4f} @ epoch {best_epoch}")
                break
        
        training_history[f'fold_{fold_id}'] = fold_history
        
        print(f"\n🧪 Testing fold {fold_id} using best model from epoch {best_epoch}...")
        
        if best_model_state:
            model.load_state_dict(best_model_state)
            
            best_model_path = os.path.join(save_dir, f'best_model_fold_{fold_id}.ckpt')
            torch.save(best_model_state, best_model_path)
            print(f"💾 Saved best model for fold {fold_id}: {best_model_path}")
        else:
            print("⚠️  Warning: No best model state found. Using the last model.")

        # 在测试集上评估
        y_true, y_pred, test_acc, test_weighted_acc, test_f1 = test_fold_model(model, test_loader, device)
        all_test_true_labels.extend(y_true)
        all_test_predictions.extend(y_pred)
        
        print(f"\n📊 Fold {fold_id} Test Results:")
        print(f"   ✅ Test Accuracy: {test_acc:.4f}")
        print(f"   ⚖️  Test Weighted Accuracy: {test_weighted_acc:.4f}")
        print(f"   🎯 Test Macro F1-score: {test_f1:.4f}")
        
        fold_results.append(test_acc)
        fold_weighted_results.append(test_weighted_acc)
        fold_f1_results.append(test_f1)
        
        # 生成测试集分类报告
        report = classification_report(
            y_true, y_pred, 
            labels=list(range(num_classes)), 
            target_names=list(label_dict.keys()), 
            digits=4
        )
        print(f"\n📋 Test Classification Report (Fold {fold_id}):")
        print(report)
        
        report_path = os.path.join(save_dir, f'test_classification_report_fold_{fold_id}.txt')
        with open(report_path, 'w') as f:
            f.write(f"Fold {fold_id} Test Results:\n")
            f.write(f"Test Accuracy: {test_acc:.4f}\n")
            f.write(f"Test Weighted Accuracy: {test_weighted_acc:.4f}\n")
            f.write(f"Test Macro F1-score: {test_f1:.4f}\n\n")
            f.write("Test Classification Report:\n")
            f.write(report)
        
        # 生成测试集混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=list(label_dict.keys()), 
                    yticklabels=list(label_dict.keys()))
        plt.title(f'Test Confusion Matrix - Fold {fold_id}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        cm_path = os.path.join(save_dir, f'test_confusion_matrix_fold_{fold_id}.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved classification report: {report_path}")
        print(f"💾 Saved confusion matrix: {cm_path}")
        
    print("\n" + "=" * 80)
    print("5-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    mean_weighted_acc = np.mean(fold_weighted_results)
    std_weighted_acc = np.std(fold_weighted_results)
    mean_f1 = np.mean(fold_f1_results)
    std_f1 = np.std(fold_f1_results)
    
    print(f"\n📊 Average Test Performance:")
    print(f"   ✅ Test Accuracy: {mean_acc:.4f} (±{std_acc:.4f})")
    print(f"   ⚖️  Test Weighted Accuracy: {mean_weighted_acc:.4f} (±{std_weighted_acc:.4f})")
    print(f"   🎯 Test Macro F1-score: {mean_f1:.4f} (±{std_f1:.4f})")
    
    print(f"\n📋 Per-Fold Test Results:")
    print(f"   {'Fold':<6} {'Acc (%)':<8} {'W-Acc (%)':<10} {'F1 (%)':<8}")
    print(f"   {'-'*6} {'-'*8} {'-'*10} {'-'*8}")
    for i in range(5):
        print(f"   {i+1:<6} {fold_results[i]*100:<8.2f} {fold_weighted_results[i]*100:<10.2f} {fold_f1_results[i]*100:<8.2f}")
    
    print(f"\n🏆 Best and Worst Folds:")
    print(f"   🥇 Best Test Acc:     Fold {np.argmax(fold_results)+1} ({max(fold_results)*100:.2f}%)")
    print(f"   🥉 Worst Test Acc:    Fold {np.argmin(fold_results)+1} ({min(fold_results)*100:.2f}%)")
    print(f"   🥇 Best Test W-Acc:   Fold {np.argmax(fold_weighted_results)+1} ({max(fold_weighted_results)*100:.2f}%)")
    print(f"   🥇 Best Test F1:      Fold {np.argmax(fold_f1_results)+1} ({max(fold_f1_results)*100:.2f}%)")
    
    # 整体测试性能报告
    print(f"\n📋 Overall Test Performance (Aggregated across all folds):")
    overall_test_report = classification_report(
        all_test_true_labels, all_test_predictions,
        labels=list(range(num_classes)), 
        target_names=list(label_dict.keys()), 
        digits=4
    )
    print(overall_test_report)
    
    # 保存汇总结果
    summary_results = {
        'mean_test_accuracy': mean_acc,
        'std_test_accuracy': std_acc,
        'mean_test_weighted_accuracy': mean_weighted_acc,
        'std_test_weighted_accuracy': std_weighted_acc,
        'mean_test_f1': mean_f1,
        'std_test_f1': std_f1,
        'fold_test_accuracies': fold_results,
        'fold_test_weighted_accuracies': fold_weighted_results,
        'fold_test_f1_scores': fold_f1_results,
        'best_fold_accuracy': int(np.argmax(fold_results)) + 1,
        'best_test_accuracy': float(max(fold_results)),
        'worst_fold_accuracy': int(np.argmin(fold_results)) + 1,
        'worst_test_accuracy': float(min(fold_results))
    }
    
    overall_summary_path = os.path.join(save_dir, 'overall_test_summary.txt')
    with open(overall_summary_path, 'w') as f:
        f.write("5-Fold Cross-Validation Test Summary:\n")
        f.write(f"Average Test Accuracy: {mean_acc:.4f} (±{std_acc:.4f})\n")
        f.write(f"Average Test Weighted Accuracy: {mean_weighted_acc:.4f} (±{std_weighted_acc:.4f})\n")
        f.write(f"Average Test Macro F1-score: {mean_f1:.4f} (±{std_f1:.4f})\n\n")
        f.write("Overall Test Classification Report (aggregated over all folds):\n")
        f.write(overall_test_report)
        
    results_json_path = os.path.join(save_dir, 'test_results.json')
    with open(results_json_path, 'w') as f:
        json.dump(summary_results, f, indent=4)
        
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=4)
        
    plot_training_history(training_history, save_dir, dataset_name)
    
    print(f"\n💾 All results saved in: {save_dir}")
    print(f"   📄 Test summary: {overall_summary_path}")
    print(f"   📊 Test results JSON: {results_json_path}")
    print(f"   📈 Training history: {history_path}")
    print(f"   📊 Training plots: {save_dir}/training_history_fold_*.png")

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch，返回loss和acc"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch in train_loader:
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
        
        del feats, padding_mask, labels, outputs, loss
    
    avg_loss = total_loss / len(train_loader)
    accuracy = (total_correct / total_samples) if total_samples > 0 else 0
    return avg_loss, accuracy

def validate_one_epoch(model, val_loader, criterion, device):
    """验证一个epoch，返回loss和各种acc/f1指标"""
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
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    weighted_accuracy = balanced_accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    return avg_loss, accuracy, weighted_accuracy, f1

def test_fold_model(model, test_loader, device):
    """在测试集上评估模型，返回真实标签、预测标签和性能指标"""
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
            
    accuracy = accuracy_score(all_labels, all_predictions)
    weighted_accuracy = balanced_accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    return all_labels, all_predictions, accuracy, weighted_accuracy, f1

def plot_training_history(training_history, save_dir, dataset_name='iemocap'):
    """
    绘制并保存每个fold的训练/验证曲线图
    """
    num_folds = len(training_history)
    
    for i in range(num_folds):
        fold_id = i + 1
        history = training_history[f'fold_{fold_id}']
        epochs = history['epochs']
        
        plt.figure(figsize=(18, 6))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
        plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='o')
        plt.title(f'Loss vs. Epochs (Fold {fold_id})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['train_acc'], label='Train Accuracy', marker='o')
        plt.plot(epochs, history['val_acc'], label='Validation Accuracy', marker='o')
        plt.title(f'Accuracy vs. Epochs (Fold {fold_id})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.suptitle(f'{dataset_name.upper()} Training History - Fold {fold_id}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        plot_path = os.path.join(save_dir, f'training_history_fold_{fold_id}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    from config import TrainingConfig
    
    torch.manual_seed(TrainingConfig.RANDOM_SEED)
    np.random.seed(TrainingConfig.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TrainingConfig.RANDOM_SEED)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_name = getattr(TrainingConfig, 'DATASET_NAME', 'iemocap')
    cv_type = "5fold_3train_1val_1test"
    
    run_name = f"{dataset_name}_{cv_type}_{timestamp}"
    TrainingConfig.SAVE_DIR = os.path.join("train_for_clean_models", run_name)
    
    train_with_early_stopping(TrainingConfig) 