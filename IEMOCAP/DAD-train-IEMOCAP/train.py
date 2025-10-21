#!/usr/bin/env python3
"""
IEMOCAP跨域训练脚本 (DACP+ECDA 增强版)
从预训练的IEMOCAP干净权重开始，在不同噪声环境下进行跨域测试。
- 实现 DACP 和 ECDA 新方法
- 自动识别噪声dB
- 结构化保存结果
- 生成详细报告
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, balanced_accuracy_score, precision_recall_fscore_support
)
import json
from datetime import datetime
import re

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入配置和模块
import config as cfg
from model import SSRLModel
from dataload_clean import get_cv_dataloaders as get_cv_dataloaders_clean
from dataload_noisy import get_cv_dataloaders_noisy
from utils import (
    DataAugmentation, 
    DACPManager, # 导入新模块
    ECDALoss,    # 导入新模块
)

class IEMOCAPCrossDomainTrainer:
    """
    IEMOCAP跨域训练器 (DACP+ECDA 增强版)
    从预训练的干净权重开始，在噪声环境下进行跨域适应
    """
    
    def __init__(self, fold=0, experiment_name=None):
        """初始化跨域训练器"""
        self.experiment_name = experiment_name
        log_exp_name = f" ({self.experiment_name})" if self.experiment_name else ""
        logger.info(f"🚀 初始化IEMOCAP跨域训练器{log_exp_name} (Fold {fold+1}/5)...")
        
        # 基本配置
        self.fold = fold
        self.device = torch.device("cuda" if torch.cuda.is_available() and cfg.USE_CUDA else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 提取噪声信息（root类型、噪声类型、噪声强度）
        self.noise_info = self._extract_noise_info()
        self.results_dir = self._setup_results_directory(self.experiment_name)
        
        # SSRL训练配置
        self.WARMUP_EPOCHS = cfg.WARMUP_EPOCHS
        
        # 新的ECDA权重
        self.target_ecda_weight = cfg.WEIGHT_ECDA
        self.weight_ecda = 0.0
        
        # 渐进式权重
        self.initial_consistency_weight = cfg.INITIAL_CONSISTENCY_WEIGHT if cfg.PROGRESSIVE_TRAINING else cfg.WEIGHT_CONSISTENCY
        self.final_consistency_weight = cfg.FINAL_CONSISTENCY_WEIGHT if cfg.PROGRESSIVE_TRAINING else cfg.WEIGHT_CONSISTENCY
        self.current_consistency_weight = self.initial_consistency_weight
        
        # 增强的性能跟踪
        self.best_clean_acc = 0.0
        self.best_noisy_acc = 0.0
        self.best_clean_weighted_f1 = 0.0
        self.best_noisy_weighted_f1 = 0.0
        self.best_clean_weighted_acc = 0.0
        self.best_noisy_weighted_acc = 0.0
        
        self.best_results = {
            'epoch': 0, 'clean_results': None, 'noisy_results': None,
            'clean_confusion_matrix': None, 'noisy_confusion_matrix': None
        }
        
        self.training_history = defaultdict(list)
        self.patience_counter = 0
        
        # [分析工具] 确认偏差追踪初始化
        self.tracked_sample_indices = None
        self.bias_analysis_log = []
        
        # [新增] 初始化 DACP 管理器和锚点
        self.dacp_manager = None
        self.calibrated_anchors = None
        
        logger.info("📋 IEMOCAP跨域训练配置 (DACP+ECDA 增强版):")
        logger.info(f"    - 数据源: IEMOCAP干净数据 → IEMOCAP噪声数据")
        logger.info(f"    - 噪声配置: {self.noise_info['display_name']}")
        logger.info(f"    - 预训练权重: {cfg.PRETRAINED_IEMOCAP_WEIGHT}")
        logger.info(f"    - 结果目录: {self.results_dir}")
        logger.info(f"    - 交叉验证: Fold {fold+1}/5")
        logger.info(f"    - 预热阶段: {self.WARMUP_EPOCHS} epochs")
        logger.info(f"    - 核心监控指标: 噪声域加权准确率")
        
        # 初始化组件
        self._setup_components()

    def _extract_noise_info(self):
        """从噪声数据路径中提取噪声信息（root类型、噪声类型、噪声强度）"""
        noisy_path = cfg.NOISY_DATA_DIR
        logger.info(f"🔍 解析噪声路径: {noisy_path}")
        
        # 提取完整的路径分析
        import re
        
        # 尝试匹配多种Root1格式和Root2格式
        root1_pattern_with_wav = r'root1-([^.]+)\.wav-(\d+)db'  # root1-babble.wav-20db
        root1_pattern_without_wav = r'root1-([^-]+)-(\d+)db'    # root1-babble-20db
        root2_pattern = r'root2-(\d+)db'                        # root2-20db
        
        # 首先尝试root1格式（带.wav）
        root1_match = re.search(root1_pattern_with_wav, noisy_path, re.IGNORECASE)
        if root1_match:
            noise_type = root1_match.group(1)  # babble
            db_value = root1_match.group(2)    # 20
            
            noise_info = {
                'root_type': 'root1',
                'noise_type': noise_type,
                'db_value': f"{db_value}db",
                'display_name': f"root1-{noise_type}-{db_value}db"
            }
            logger.info(f"✅ 成功识别Root1噪声（带.wav）: 类型={noise_type}, 强度={db_value}dB")
            return noise_info
        
        # 然后尝试root1格式（不带.wav）
        root1_match = re.search(root1_pattern_without_wav, noisy_path, re.IGNORECASE)
        if root1_match:
            noise_type = root1_match.group(1)  # babble
            db_value = root1_match.group(2)    # 20
            
            noise_info = {
                'root_type': 'root1',
                'noise_type': noise_type,
                'db_value': f"{db_value}db",
                'display_name': f"root1-{noise_type}-{db_value}db"
            }
            logger.info(f"✅ 成功识别Root1噪声（不带.wav）: 类型={noise_type}, 强度={db_value}dB")
            return noise_info
        
        # 尝试root2格式
        root2_match = re.search(root2_pattern, noisy_path, re.IGNORECASE)
        if root2_match:
            db_value = root2_match.group(1)    # 20
            
            noise_info = {
                'root_type': 'root2',
                'noise_type': None,  # root2不需要噪声类型
                'db_value': f"{db_value}db",
                'display_name': f"root2-{db_value}db"
            }
            logger.info(f"✅ 成功识别Root2噪声: 强度={db_value}dB")
            return noise_info
        
        # 如果都没匹配到，尝试通用db提取
        db_patterns = [r'(\d+)db', r'(-?\d+)_?db']
        for pattern in db_patterns:
            match = re.search(pattern, noisy_path, re.IGNORECASE)
            if match:
                db_value = match.group(1)
                noise_info = {
                    'root_type': 'unknown',
                    'noise_type': 'unknown',
                    'db_value': f"{db_value}db",
                    'display_name': f"unknown-{db_value}db"
                }
                logger.warning(f"⚠️ 使用通用格式识别噪声: {db_value}dB")
                return noise_info
        
        # 完全无法识别的情况
        logger.warning(f"⚠️ 无法从路径 '{noisy_path}' 中识别噪声信息，使用默认值")
        return {
            'root_type': 'unknown',
            'noise_type': 'unknown', 
            'db_value': 'unknown_db',
            'display_name': 'unknown-unknown-unknown_db'
        }

    def _setup_results_directory(self, experiment_name=None):
        """
        设置多层级结果目录结构
        
        结构：
        iemocap_mutil-noisy_cross_domain_results/
        ├── root1/
        │   ├── babble/
        │   │   ├── 20db/
        │   │   │   └── fold_X/
        │   │   └── 15db/
        │   └── white/
        └── root2/
            ├── 20db/
            │   └── fold_X/
            └── 15db/
        """
        base_dir = "iemocap_mutil-noisy_cross_domain_results"
        
        if experiment_name:
            # 如果提供了实验名称，在根目录下创建实验名称子目录
            safe_exp_name = re.sub(r'[\\/*?:"<>|]', "", experiment_name)
            base_dir = os.path.join(base_dir, safe_exp_name)
        
        # 根据root类型构建路径
        root_type = self.noise_info['root_type']
        
        if root_type == 'root1':
            # root1: base/root1/noise_type/db/fold_X
            noise_type = self.noise_info['noise_type']
            db_value = self.noise_info['db_value']
            results_dir = os.path.join(base_dir, root_type, noise_type, db_value, f"fold_{self.fold+1}")
            
        elif root_type == 'root2':
            # root2: base/root2/db/fold_X
            db_value = self.noise_info['db_value']
            results_dir = os.path.join(base_dir, root_type, db_value, f"fold_{self.fold+1}")
            
        else:
            # unknown: base/unknown/fold_X
            results_dir = os.path.join(base_dir, "unknown", f"fold_{self.fold+1}")
        
        # 创建目录结构
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "reports"), exist_ok=True)
        
        logger.info(f"📁 多层级结果目录已创建: {results_dir}")
        logger.info(f"📂 目录结构: {self.noise_info['display_name']} → fold_{self.fold+1}")
        
        return results_dir
    
    def _setup_components(self):
        """设置所有训练组件"""
        self._setup_dataloaders()
        self._setup_model()
        self._load_pretrained_weights()
        
        # [新增] 运行一次性的锚点校准
        if cfg.ANCHOR_CALIBRATION_ENABLED:
            self._run_anchor_calibration()
        
        self._setup_training_components()
        self._setup_augmentation()
    
    def _setup_dataloaders(self):
        """设置数据加载器"""
        logger.info("🗂️ 设置IEMOCAP数据加载器...")
        clean_data_path = cfg.CLEAN_DATA_DIR
        noisy_data_path = cfg.NOISY_DATA_DIR
        fold_id = self.fold + 1

        try:
            self.clean_train_loader, self.clean_val_loader, self.clean_test_loader, self.class_names, self.num_classes = \
                get_cv_dataloaders_clean(clean_data_path, cfg.BATCH_SIZE, fold_id=fold_id)
            
            self.noisy_student_loader, self.noisy_teacher_loader, self.noisy_val_loader, self.noisy_test_loader = \
                get_cv_dataloaders_noisy(noisy_data_path, cfg.BATCH_SIZE, fold_id=fold_id)
            
            logger.info(f"✅ IEMOCAP数据加载器设置完成 (Fold {fold_id}/5)")
            logger.info(f"   🧹 干净数据 - 训练: {len(self.clean_train_loader)} 批次, 验证: {len(self.clean_val_loader)} 批次, 测试: {len(self.clean_test_loader)} 批次")
            logger.info(f"   🔇 噪声数据 - 训练: {len(self.noisy_student_loader)} 批次, 验证: {len(self.noisy_val_loader)} 批次, 测试: {len(self.noisy_test_loader)} 批次")
            
            # [分析工具] 随机选择追踪样本进行确认偏差分析
            num_track_samples = 50
            total_noisy_samples = len(self.noisy_student_loader.dataset)
            if total_noisy_samples > num_track_samples:
                self.tracked_sample_indices = np.random.choice(
                    total_noisy_samples, num_track_samples, replace=False
                ).tolist()
                logger.info(f"🔬 将追踪 {num_track_samples} 个样本以分析确认偏差")
            
        except Exception as e:
            logger.error(f"❌ 数据加载器设置失败: {e}")
            raise
    
    def _setup_model(self):
        """设置模型"""
        logger.info("🏗️ 设置IEMOCAP跨域SSRL模型...")
        self.model = SSRLModel(cfg=cfg).to(self.device)
        if cfg.PRINT_MODEL_INFO:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"📊 模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
        logger.info("✅ IEMOCAP跨域SSRL模型设置完成")

    def _load_pretrained_weights(self):
        """加载预训练权重"""
        logger.info(f"💾 加载IEMOCAP预训练权重: {cfg.PRETRAINED_IEMOCAP_WEIGHT}")
        if not os.path.exists(cfg.PRETRAINED_IEMOCAP_WEIGHT):
            logger.error(f"❌ 预训练权重文件不存在!")
            raise FileNotFoundError(f"预训练权重文件不存在: {cfg.PRETRAINED_IEMOCAP_WEIGHT}")
        
        try:
            self.model.load_complete_pretrained_weights(cfg.PRETRAINED_IEMOCAP_WEIGHT)
            self.model._init_teacher_network()
            logger.info("✅ 成功将预训练权重加载到学生网络并同步到教师网络。")
            
        except Exception as e:
            logger.error(f"❌ 加载预训练权重失败: {e}")
            raise

    def _run_anchor_calibration(self):
        """[新增] 执行一次性的锚点校准"""
        logger.info("⚓️ 正在执行一次性锚点校准 (Anchor Calibration)...")
        clean_data_path = cfg.CLEAN_DATA_DIR
        noisy_data_path = cfg.NOISY_DATA_DIR
        fold_id = self.fold + 1

        clean_calib_loader, _, _, _, _ = get_cv_dataloaders_clean(clean_data_path, cfg.BATCH_SIZE * 2, fold_id=fold_id)
        _, _, noisy_calib_loader, _ = get_cv_dataloaders_noisy(noisy_data_path, cfg.BATCH_SIZE * 2, fold_id=fold_id)

        self.model.eval()
        scores_per_class = {'clean': [[] for _ in range(self.num_classes)], 'noisy': [[] for _ in range(self.num_classes)]}

        with torch.no_grad():
            for batch in clean_calib_loader:
                feats, mask, labels = batch['net_input']['feats'].to(self.device), batch['net_input']['padding_mask'].to(self.device), batch['labels'].to(self.device)
                probs = F.softmax(self.model.predict(feats, mask), dim=1)
                scores, _ = DACPManager.calculate_certainty_scores(probs)
                for i, label in enumerate(labels):
                    scores_per_class['clean'][label.item()].append(scores[i].item())
            
            for batch in noisy_calib_loader:
                feats, mask, labels = batch['net_input']['feats'].to(self.device), batch['net_input']['padding_mask'].to(self.device), batch['labels'].to(self.device)
                probs = F.softmax(self.model.predict(feats, mask), dim=1)
                scores, _ = DACPManager.calculate_certainty_scores(probs)
                for i, label in enumerate(labels):
                    scores_per_class['noisy'][label.item()].append(scores[i].item())
        
        avg_scores_clean = torch.tensor([np.mean(s) if s else 0 for s in scores_per_class['clean']], device=self.device)
        avg_scores_noisy = torch.tensor([np.mean(s) if s else 0 for s in scores_per_class['noisy']], device=self.device)
        shift_ratio = avg_scores_noisy / (avg_scores_clean + 1e-8)
        mu_clean = avg_scores_clean
        sigma_clean = torch.tensor([np.std(s) if s else 0 for s in scores_per_class['clean']], device=self.device)
        base_anchor = torch.clamp(mu_clean - cfg.ANCHOR_STD_K * sigma_clean, min=0)
        self.calibrated_anchors = base_anchor * shift_ratio
        
        logger.info("✅ 锚点校准完成!")
        for c, anchor in enumerate(self.calibrated_anchors):
            logger.info(f"   - 类别 '{self.class_names[c]}': 校准锚点 τ_calibrated_anchor = {anchor:.4f}")
        
        self.model.train()

    def _setup_training_components(self):
        """设置训练组件"""
        logger.info("⚙️ 设置跨域训练组件 (DACP+ECDA)...")
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.EPOCHS) if cfg.LEARNING_RATE_SCHEDULER == "cosine" else None
        self.ce_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING_FACTOR if cfg.USE_LABEL_SMOOTHING else 0.0)
        self.kl_criterion = nn.KLDivLoss(reduction='none') # 'none'以便手动加权
        
        self.dacp_manager = DACPManager(self.num_classes, cfg.EPOCHS, self.device)
        self.ecda_criterion = ECDALoss()
        
        logger.info(f"✅ 跨域训练组件设置完成 (学习率: {cfg.LEARNING_RATE})")

    def _setup_augmentation(self):
        """设置数据增强器"""
        self.augmenter = DataAugmentation()
        logger.info("✅ 数据增强器设置完成")

    def is_warmup_phase(self, epoch):
        return epoch < self.WARMUP_EPOCHS

    def update_loss_weights(self, epoch):
        if self.is_warmup_phase(epoch):
            self.weight_ecda, self.current_consistency_weight = 0.0, 0.0
            return
        
        if cfg.PROGRESSIVE_TRAINING:
            progress = min(1.0, (epoch - self.WARMUP_EPOCHS) / cfg.WEIGHT_RAMP_EPOCHS)
            self.current_consistency_weight = self.initial_consistency_weight + (self.final_consistency_weight - self.initial_consistency_weight) * progress
        else:
            self.current_consistency_weight = cfg.WEIGHT_CONSISTENCY
        
        if epoch >= cfg.ECDA_START_EPOCH:
            ecda_progress = min(1.0, (epoch - cfg.ECDA_START_EPOCH) / cfg.WEIGHT_RAMP_EPOCHS)
            self.weight_ecda = cfg.WEIGHT_ECDA * ecda_progress
        else:
            self.weight_ecda = 0.0

    def train_step(self, clean_batch, noisy_batch, epoch):
        clean_feats, clean_padding_mask, clean_labels = clean_batch['net_input']['feats'].to(self.device), clean_batch['net_input']['padding_mask'].to(self.device), clean_batch['labels'].to(self.device)
        student_clean_encoded = self.model.student_encoder(clean_feats, clean_padding_mask)
        supervised_ce_loss = self.ce_criterion(self.model.student_classifier(student_clean_encoded), clean_labels)
        
        if self.is_warmup_phase(epoch):
            return {'total_loss': supervised_ce_loss, 'supervised_ce_loss': supervised_ce_loss, 'consistency_loss': torch.tensor(0.0, device=self.device), 'ecda_loss': torch.tensor(0.0, device=self.device)}
        
        noisy_feats, noisy_padding_mask = noisy_batch['net_input']['feats'].to(self.device), noisy_batch['net_input']['padding_mask'].to(self.device)
        noisy_weak, noisy_strong = self.augmenter.weak_augment(noisy_feats), self.augmenter.strong_augment(noisy_feats)
        
        with torch.no_grad():
            teacher_encoded = self.model.teacher_encoder(noisy_weak, noisy_padding_mask)
            teacher_probs = F.softmax(self.model.teacher_classifier(teacher_encoded), dim=1)
        
        # [新增] 根据配置选择DACP或固定阈值
        if cfg.USE_DACP:
            mask, certainty_scores, class_weights_wce = self.dacp_manager.calculate_mask(
                teacher_probs, epoch, self.calibrated_anchors
            )
        else:
            certainty_scores, _ = torch.max(teacher_probs, dim=1)
            mask = (certainty_scores >= cfg.FIXED_CONFIDENCE_THRESHOLD).float()
            class_weights_wce = torch.ones_like(mask)

        high_confidence_count = mask.sum().item()
        
        # [分析工具] 记录确认偏差追踪数据
        if self.tracked_sample_indices and 'id' in noisy_batch:
            _, preds = torch.max(teacher_probs, dim=1)
            batch_indices = noisy_batch['id'].cpu().numpy()
            for i, sample_idx in enumerate(batch_indices):
                if int(sample_idx) in self.tracked_sample_indices:
                    log_entry = {
                        'epoch': epoch,
                        'sample_id': int(sample_idx),
                        'pseudo_label': int(preds[i].item()),
                        'certainty_score': float(certainty_scores[i].item()),
                        'is_masked_in': bool(mask[i].item())
                    }
                    self.bias_analysis_log.append(log_entry)
        
        student_strong_encoded = self.model.student_encoder(noisy_strong, noisy_padding_mask)
        student_log_probs = F.log_softmax(self.model.student_classifier(student_strong_encoded), dim=1)
        
        consistency_loss, ecda_loss = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        if high_confidence_count > 1:
            kl_div = self.kl_criterion(student_log_probs, teacher_probs)
            kl_div_per_sample = kl_div.sum(dim=1)
            consistency_loss = (kl_div_per_sample * mask).sum() / (mask.sum() + 1e-8)
            
            # [新增] 根据配置决定是否计算ECDA损失
            if cfg.USE_ECDA and self.weight_ecda > 0:
                _, noisy_pseudo_labels = torch.max(teacher_probs, dim=1)
                ecda_loss = self.ecda_criterion(
                    clean_feats=student_clean_encoded,
                    noisy_feats=student_strong_encoded,
                    clean_labels=clean_labels,
                    noisy_labels=noisy_pseudo_labels,
                    noisy_mask=mask,
                    noisy_scores=certainty_scores,
                    class_weights_wce=class_weights_wce
                )

        total_loss = (
            supervised_ce_loss + 
            self.current_consistency_weight * consistency_loss +
            self.weight_ecda * ecda_loss
        )
        
        return {
            'total_loss': total_loss, 'supervised_ce_loss': supervised_ce_loss, 
            'consistency_loss': consistency_loss, 'ecda_loss': ecda_loss
        }

    def train_epoch(self, epoch):
        self.model.train()
        self.update_loss_weights(epoch)
        total_losses = defaultdict(float)
        num_batches = 0
        
        clean_iter, noisy_student_iter = iter(self.clean_train_loader), iter(self.noisy_student_loader)
        max_batches = min(len(self.clean_train_loader), len(self.noisy_student_loader))

        for batch_idx in range(max_batches):
            clean_batch, noisy_batch = next(clean_iter), next(noisy_student_iter)
            self.optimizer.zero_grad()
            losses = self.train_step(clean_batch, noisy_batch, epoch)
            losses['total_loss'].backward()
            if cfg.GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.MAX_GRAD_NORM)
            self.optimizer.step()
            
            if not self.is_warmup_phase(epoch):
                self.model.update_teacher_ema()
                
            for key, value in losses.items():
                if isinstance(value, torch.Tensor): total_losses[key] += value.item()
            num_batches += 1
        
        if not self.is_warmup_phase(epoch):
            self.dacp_manager.update_class_quality_scores_epoch(self.dacp_manager.batch_scores_per_class)
            
            # [分析工具] 记录DACP状态和ECDA注意力权重
            self.training_history['dacp_ema_thresholds'].append(
                self.dacp_manager.ema_thresholds.cpu().numpy().tolist()
            )
            self.training_history['dacp_class_quality'].append(
                self.dacp_manager.class_quality_scores.cpu().numpy().tolist()
            )
            
            # 计算并记录ECDA注意力权重
            class_weights = self.dacp_manager.class_quality_scores
            avg_class_weight = class_weights.mean()
            class_attention_weights = torch.exp(
                cfg.ECDA_CLASS_ATTENTION_LAMBDA * (avg_class_weight - class_weights)
            )
            self.training_history['ecda_class_attention'].append(
                class_attention_weights.cpu().numpy().tolist()
            )
        
        if self.scheduler: self.scheduler.step()
        return {key: value / num_batches for key, value in total_losses.items()}

    def validate(self, data_loader, domain_name="unknown"):
        """验证函数 - 计算详细评估指标"""
        self.model.eval()
        all_preds_student, all_labels = [], []
        with torch.no_grad():
            for batch in data_loader:
                feats, mask, labels = batch['net_input']['feats'].to(self.device), batch['net_input']['padding_mask'].to(self.device), batch['labels'].to(self.device)
                outputs = self.model.predict(feats, mask, use_teacher=False)
                _, predicted = torch.max(outputs, 1)
                all_preds_student.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # [分析工具] 计算教师-学生网络差异性
        if "noisy" in domain_name.lower() and not self.is_warmup_phase(getattr(self, 'current_epoch', 0)):
            all_preds_teacher = []
            with torch.no_grad():
                for batch in data_loader:
                    feats, mask = batch['net_input']['feats'].to(self.device), batch['net_input']['padding_mask'].to(self.device)
                    outputs = self.model.predict(feats, mask, use_teacher=True)
                    _, predicted = torch.max(outputs, 1)
                    all_preds_teacher.extend(predicted.cpu().numpy())
            
            # 计算差异率
            if len(all_preds_student) == len(all_preds_teacher):
                disagreement_rate = np.mean(
                    np.array(all_preds_student) != np.array(all_preds_teacher)
                )
                self.training_history[f'disagreement_rate_{domain_name.lower()}'].append(
                    disagreement_rate
                )
        
        cm = confusion_matrix(all_labels, all_preds_student, labels=range(self.num_classes))
        (prec, rec, f1, sup) = precision_recall_fscore_support(all_labels, all_preds_student, average=None, zero_division=0, labels=range(self.num_classes))

        return {
            'accuracy': accuracy_score(all_labels, all_preds_student) * 100,
            'weighted_accuracy': balanced_accuracy_score(all_labels, all_preds_student) * 100,
            'f1_weighted': f1_score(all_labels, all_preds_student, average='weighted', zero_division=0) * 100,
            'f1_macro': f1_score(all_labels, all_preds_student, average='macro', zero_division=0) * 100,
            'precision_per_class': prec.tolist(), 'recall_per_class': rec.tolist(),
            'f1_per_class': f1.tolist(), 'support_per_class': sup.tolist(),
            'confusion_matrix': cm
        }

    def check_early_stopping(self, noisy_results, is_best):
        """早停检查 - 核心监控噪声域加权准确率"""
        if not cfg.EARLY_STOPPING: return False
        if is_best:
            self.patience_counter = 0
            logger.info(f"🎯 发现更佳模型! 噪声域加权准确率: {noisy_results['weighted_accuracy']:.2f}% - 早停计数器重置")
            return False
        else:
            self.patience_counter += 1
            logger.info(f"⏳ 早停计数: {self.patience_counter}/{cfg.PATIENCE} (当前噪声域加权准确率: {noisy_results['weighted_accuracy']:.2f}%)")
            if self.patience_counter >= cfg.PATIENCE:
                logger.info(f"🛑 早停触发!")
                return True
        return False

    def save_checkpoint(self, epoch, clean_results, noisy_results, is_best=False):
        """保存检查点"""
        checkpoint = {'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'clean_results': clean_results, 'noisy_results': noisy_results}
        models_dir = os.path.join(self.results_dir, "models")
        
        if is_best:
            self.best_results.update({'epoch': epoch, 'clean_results': clean_results, 'noisy_results': noisy_results})
            best_path = os.path.join(models_dir, f'iemocap_cross_domain_best.pth')
            torch.save(checkpoint, best_path)
            self._save_confusion_matrices(clean_results, noisy_results, epoch, is_best_result=True)
            self._save_detailed_results(clean_results, noisy_results, epoch, is_best_result=True)
            logger.info(f"💾 Best model saved: {best_path}")

    def _save_confusion_matrices(self, clean_results, noisy_results, epoch, is_best_result=False):
        """生成并保存混淆矩阵图"""
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        sns.heatmap(clean_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names, ax=ax1)
        ax1.set_title(f'Clean Domain (Epoch {epoch+1})\nAcc: {clean_results["accuracy"]:.2f}%, W-Acc: {clean_results["weighted_accuracy"]:.2f}%')
        sns.heatmap(noisy_results['confusion_matrix'], annot=True, fmt='d', cmap='Oranges', xticklabels=self.class_names, yticklabels=self.class_names, ax=ax2)
        ax2.set_title(f'Noisy Domain ({self.noise_info["display_name"]}) (Epoch {epoch+1})\nAcc: {noisy_results["accuracy"]:.2f}%, W-Acc: {noisy_results["weighted_accuracy"]:.2f}%')
        if is_best_result: fig.suptitle('🏆 BEST RESULTS 🏆', fontsize=16, weight='bold')
        plots_dir = os.path.join(self.results_dir, "plots")
        filename = f'BEST_confusion_matrices_epoch_{epoch+1}.png' if is_best_result else f'confusion_matrices_epoch_{epoch+1}.png'
        plt.savefig(os.path.join(plots_dir, filename), dpi=300)
        plt.close()

    def _save_detailed_results(self, clean_results, noisy_results, epoch, is_best_result=False):
        """保存详细JSON结果报告"""
        def _calculate_per_class_accuracy(cm):
            return [(cm[i, i] / cm[i, :].sum()) if cm[i, :].sum() > 0 else 0.0 for i in range(len(cm))]
        
        results_summary = {
            'info': {
                'noise_config': self.noise_info,
                'fold': self.fold + 1, 
                'epoch': epoch + 1, 
                'is_best': is_best_result
            },
            'summary': {
                'clean': {'acc': f"{clean_results['accuracy']:.2f}%", 'w_acc': f"{clean_results['weighted_accuracy']:.2f}%", 'w_f1': f"{clean_results['f1_weighted']:.2f}%"},
                'noisy': {'acc': f"{noisy_results['accuracy']:.2f}%", 'w_acc': f"{noisy_results['weighted_accuracy']:.2f}%", 'w_f1': f"{noisy_results['f1_weighted']:.2f}%"}
            },
            'details': {
                'class_names': self.class_names,
                'clean': {'precision': clean_results['precision_per_class'], 'recall': clean_results['recall_per_class'], 'f1': clean_results['f1_per_class'], 'support': clean_results['support_per_class'], 'accuracy': _calculate_per_class_accuracy(clean_results['confusion_matrix'])},
                'noisy': {'precision': noisy_results['precision_per_class'], 'recall': noisy_results['recall_per_class'], 'f1': noisy_results['f1_per_class'], 'support': noisy_results['support_per_class'], 'accuracy': _calculate_per_class_accuracy(noisy_results['confusion_matrix'])}
            }
        }
        reports_dir = os.path.join(self.results_dir, "reports")
        filename = f'BEST_detailed_results_epoch_{epoch+1}.json' if is_best_result else f'detailed_results_epoch_{epoch+1}.json'
        with open(os.path.join(reports_dir, filename), 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=4, ensure_ascii=False)

    def train(self):
        """主训练循环"""
        logger.info(f"🚀 开始IEMOCAP跨域训练 (Fold {self.fold+1}/5, 噪声: {self.noise_info['display_name']})...")
        for epoch in range(cfg.EPOCHS):
            avg_losses = self.train_epoch(epoch)
            for key, value in avg_losses.items(): self.training_history[key].append(value)
            
            should_validate = (epoch + 1) % cfg.VALIDATION_INTERVAL == 0 or not self.is_warmup_phase(epoch)
            if should_validate:
                # [分析工具] 设置当前epoch供validate方法使用
                self.current_epoch = epoch
                clean_results = self.validate(self.clean_val_loader, "Clean")
                noisy_results = self.validate(self.noisy_val_loader, "Noisy")
                
                is_best = noisy_results['weighted_accuracy'] > self.best_noisy_weighted_acc + cfg.MIN_DELTA
                if is_best:
                    self.best_noisy_acc, self.best_clean_acc = noisy_results['accuracy'], clean_results['accuracy']
                    self.best_noisy_weighted_f1, self.best_clean_weighted_f1 = noisy_results['f1_weighted'], clean_results['f1_weighted']
                    self.best_noisy_weighted_acc, self.best_clean_weighted_acc = noisy_results['weighted_accuracy'], clean_results['weighted_accuracy']

                self.save_checkpoint(epoch, clean_results, noisy_results, is_best)
                
                loss_str = f"Losses: Total={avg_losses.get('total_loss', 0):.4f} CE={avg_losses.get('supervised_ce_loss', 0):.4f} KD={avg_losses.get('consistency_loss', 0):.4f} ECDA={avg_losses.get('ecda_loss', 0):.4f}"
                logger.info(f"Epoch {epoch+1}/{cfg.EPOCHS} | {loss_str} | Noisy WA: {noisy_results['weighted_accuracy']:.2f}% {'🔥' if is_best else ''}")
                
                if self.check_early_stopping(noisy_results, is_best): 
                    logger.info("🛑 早停触发，开始测试集最终评估...")
                    break
        
        # [分析工具] 保存训练历史和确认偏差日志
        self._save_analysis_data()
        
        # [重要] 确保在训练结束后进行测试集评估
        logger.info("🧪 开始测试集最终评估...")
        self._evaluate_on_test_set()
        
        logger.info(f"🎉 IEMOCAP跨域训练完成 (Fold {self.fold+1})!")
        return {'best_noisy_weighted_acc': self.best_noisy_weighted_acc, 'results_dir': self.results_dir}
    
    def _save_analysis_data(self):
        """保存分析所需的数据"""
        # 保存训练历史
        history_save_path = os.path.join(self.results_dir, "reports", "training_history.json")
        try:
            with open(history_save_path, 'w') as f:
                # 转换numpy数组为列表以便JSON序列化
                serializable_history = {}
                for k, v in self.training_history.items():
                    serializable_history[k] = [
                        item.tolist() if isinstance(item, np.ndarray) else item 
                        for item in v
                    ]
                json.dump(serializable_history, f, indent=4)
            logger.info(f"💾 训练历史已保存至: {history_save_path}")
        except Exception as e:
            logger.error(f"❌ 保存训练历史失败: {e}")
        
        # 保存确认偏差日志
        if self.bias_analysis_log:
            bias_log_path = os.path.join(self.results_dir, "reports", "confirmation_bias_log.json")
            try:
                with open(bias_log_path, 'w') as f:
                    json.dump(self.bias_analysis_log, f, indent=4)
                logger.info(f"💾 确认偏差日志已保存至: {bias_log_path}")
            except Exception as e:
                logger.error(f"❌ 保存确认偏差日志失败: {e}")

    def _evaluate_on_test_set(self):
        """加载最佳模型并在测试集上进行最终评估"""
        best_model_path = os.path.join(self.results_dir, "models", "iemocap_cross_domain_best.pth")
        if not os.path.exists(best_model_path):
            logger.warning(f"⚠️ 未找到最佳模型文件: {best_model_path}。跳过测试集评估。")
            return

        logger.info(f"💾 加载最佳模型进行测试集评估: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 评估干净域测试集
        logger.info("🧹 评估干净域测试集...")
        clean_test_results = self.validate(self.clean_test_loader, "Clean_Test")
        logger.info(f"   📊 Clean Domain Test - Acc: {clean_test_results['accuracy']:.2f}%, W-Acc: {clean_test_results['weighted_accuracy']:.2f}%, W-F1: {clean_test_results['f1_weighted']:.2f}%")

        # 评估噪声域测试集
        logger.info(f"🔇 评估噪声域测试集 ({self.noise_info['display_name']})...")
        noisy_test_results = self.validate(self.noisy_test_loader, "Noisy_Test")
        logger.info(f"   📊 Noisy Domain Test - Acc: {noisy_test_results['accuracy']:.2f}%, W-Acc: {noisy_test_results['weighted_accuracy']:.2f}%, W-F1: {noisy_test_results['f1_weighted']:.2f}%")

        # 保存测试集混淆矩阵和详细结果
        self._save_confusion_matrices(clean_test_results, noisy_test_results, 999, is_best_result=False)  # 使用999作为测试集标识
        self._save_detailed_results(clean_test_results, noisy_test_results, 999, is_best_result=False)
        
        # 生成并保存最终测试报告
        final_test_summary = {
            'info': {
                'noise_config': self.noise_info,
                'fold': self.fold + 1,
                'evaluation_type': 'Final Test Set Evaluation',
                'timestamp': datetime.now().isoformat()
            },
            'final_test_results': {
                'clean_domain': {
                    'accuracy': f"{clean_test_results['accuracy']:.2f}%",
                    'weighted_accuracy': f"{clean_test_results['weighted_accuracy']:.2f}%",
                    'weighted_f1': f"{clean_test_results['f1_weighted']:.2f}%"
                },
                'noisy_domain': {
                    'accuracy': f"{noisy_test_results['accuracy']:.2f}%",
                    'weighted_accuracy': f"{noisy_test_results['weighted_accuracy']:.2f}%",
                    'weighted_f1': f"{noisy_test_results['f1_weighted']:.2f}%"
                }
            },
            'comparison_with_validation': {
                'validation_best_noisy_weighted_acc': f"{self.best_noisy_weighted_acc:.2f}%",
                'test_noisy_weighted_acc': f"{noisy_test_results['weighted_accuracy']:.2f}%",
                'performance_gap': f"{(noisy_test_results['weighted_accuracy'] - self.best_noisy_weighted_acc):.2f}%"
            }
        }
        
        test_report_path = os.path.join(self.results_dir, "reports", "FINAL_test_set_results.json")
        with open(test_report_path, 'w', encoding='utf-8') as f:
            json.dump(final_test_summary, f, indent=4, ensure_ascii=False)
        
        logger.info(f"💾 最终测试集结果已保存至: {test_report_path}")
        logger.info("✅ 测试集评估完成!")
        
        return clean_test_results, noisy_test_results


def main():
    """主函数：运行5折交叉验证"""
    logger.info("="*50)
    logger.info("🚀 开始 IEMOCAP 跨域DACP+ECDA 5折交叉验证 🚀")
    logger.info("="*50)
    
    all_folds_results = []
    
    logger.info(f"\n{'='*20} FOLD {cfg.N_FOLDS}/{5} {'='*20}")
        
    try:
        # 每个fold重新实例化训练器
        # [修改] main函数不再负责交叉验证循环，仅运行单次
        trainer = IEMOCAPCrossDomainTrainer(fold=cfg.N_FOLDS-1)
        fold_best_results = trainer.train()
        all_folds_results.append(fold_best_results)
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"❌ Fold {cfg.N_FOLDS} 训练失败: {e}", exc_info=True)
        # 可选择在这里中断或继续下一个fold
        # continue


    logger.info("\n" + "="*50)
    logger.info("🎉 5折交叉验证全部完成 🎉")
    logger.info("="*50)
    
    # 最终报告汇总
    # final_report_path = os.path.join("iemocap_cross_domain_results", "final_summary_report.json")
    # with open(final_report_path, 'w') as f:
    #     json.dump(all_folds_results, f, indent=4)
    # logger.info(f"Final summary report saved to {final_report_path}")

if __name__ == "__main__":
    main() 