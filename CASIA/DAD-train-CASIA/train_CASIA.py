#!/usr/bin/env python3
"""
CASIA跨域训练脚本 (DACP+ECDA 增强版)
从预训练的CASIA干净权重开始，在不同噪声环境下进行跨域测试。
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

# 导入CASIA配置和模块
import config_casia as cfg
from model import SSRLModel
from dataload_casia_clean import create_casia_clean_dataloaders
from dataload_casia_noisy import create_casia_noisy_dataloaders_with_speaker_isolation
from utils import (
    DataAugmentation, 
    DACPManager, # 导入新模块
    ECDALoss,    # 导入新模块
)

class FixedCASIACrossDomainTrainer:
    """
    CASIA跨域训练器 (DACP+ECDA 增强版)
    从预训练的干净权重开始，在噪声环境下进行跨域适应
    """
    
    def __init__(self, fold=3, experiment_name=None):  # 默认使用第4折权重
        """初始化跨域训练器"""
        self.experiment_name = experiment_name
        log_exp_name = f" ({self.experiment_name})" if self.experiment_name else ""
        logger.info(f"🚀 初始化CASIA跨域训练器 (DACP+ECDA 增强版){log_exp_name} (Fold {fold+1}/4)...")
        
        # 基本配置
        self.fold = fold
        self.device = torch.device("cuda" if torch.cuda.is_available() and cfg.USE_CUDA else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 提取噪声信息（root类型、噪声类型、噪声强度）
        self.noise_info = self._extract_noise_info()
        self.results_dir = self._setup_results_directory(self.experiment_name)
        
        # SSRL训练配置
        self.WARMUP_EPOCHS = cfg.WARMUP_EPOCHS
        self.scl_start_epoch = cfg.SCL_START_EPOCH
        
        # 新的ECDA权重
        self.target_ecda_weight = cfg.WEIGHT_ECDA
        self.weight_ecda = 0.0
        
        # SCL 权重
        self.target_scl_weight = cfg.TARGET_SCL_WEIGHT
        self.weight_scl = 0.0
        
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
        
        # [新增] 初始化 DACP 管理器和锚点
        self.dacp_manager = None
        self.calibrated_anchors = None
        
        logger.info("📋 CASIA跨域训练配置 (DACP+ECDA 增强版):")
        logger.info(f"    - 数据源: CASIA干净数据 → CASIA {self.noise_info['display_name']} 噪声数据")
        logger.info(f"    - 预训练权重: {cfg.PRETRAINED_CASIA_WEIGHT}")
        logger.info(f"    - 结果目录: {self.results_dir}")
        logger.info(f"    - 交叉验证: Fold {fold+1}/4")
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
        
        # 尝试匹配多种格式模式
        processed_features_pattern = r'processed_features_([^_]+)_(\d+)db'  # processed_features_babble_20db
        root1_pattern_with_wav = r'root1-([^.]+)\.wav-(\d+)db'            # root1-babble.wav-20db
        root1_pattern_without_wav = r'root1-([^-]+)-(\d+)db'              # root1-babble-20db
        root2_pattern = r'root2-(\d+)db'                                  # root2-20db
        
        # 首先尝试新的 processed_features 格式（预处理脚本生成的格式）
        processed_match = re.search(processed_features_pattern, noisy_path, re.IGNORECASE)
        if processed_match:
            noise_type = processed_match.group(1)  # babble, factory1, hfchannel, volvo
            db_value = processed_match.group(2)    # 20, 15, 10, 5, 0
            
            noise_info = {
                'root_type': 'real_noise',
                'noise_type': noise_type,
                'db_value': f"{db_value}db",
                'display_name': f"real-{noise_type}-{db_value}db"
            }
            logger.info(f"✅ 成功识别真实噪声: 类型={noise_type}, 强度={db_value}dB")
            return noise_info
        
        # 然后尝试root1格式（带.wav）
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
        casia_cross_domain_results/
        ├── real_noise/
        │   ├── babble/
        │   │   ├── 20db/
        │   │   │   └── fold_X/
        │   │   └── 15db/
        │   ├── factory1/
        │   ├── hfchannel/
        │   └── volvo/
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
        base_dir = "casia_mutil-noise_cross_domain_results"
        
        if experiment_name:
            # 如果提供了实验名称，在根目录下创建实验名称子目录
            safe_exp_name = re.sub(r'[\\/*?:"<>|]', "", experiment_name)
            base_dir = os.path.join(base_dir, safe_exp_name)
        
        # 根据root类型构建路径
        root_type = self.noise_info['root_type']
        
        if root_type == 'real_noise':
            # real_noise: base/real_noise/noise_type/db/fold_X
            noise_type = self.noise_info['noise_type']
            db_value = self.noise_info['db_value']
            results_dir = os.path.join(base_dir, root_type, noise_type, db_value, f"fold_{self.fold+1}")
            
        elif root_type == 'root1':
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
        logger.info("🗂️ 设置CASIA数据加载器...")
        try:
            self.clean_train_loader, self.clean_val_loader, self.clean_test_loader, self.num_classes, self.class_names = \
                create_casia_clean_dataloaders(cfg.CLEAN_FEAT_PATH, cfg.BATCH_SIZE, fold=self.fold)
            
            self.noisy_student_loader, self.noisy_teacher_loader, self.noisy_val_loader, self.noisy_test_loader = \
                create_casia_noisy_dataloaders_with_speaker_isolation(
                    cfg.NOISY_FEAT_PATH, cfg.BATCH_SIZE, fold=self.fold
                )
            
            logger.info(f"✅ CASIA数据加载器设置完成 (Fold {self.fold+1}/4)")
            logger.info(f"   🧹 干净数据 - 训练: {len(self.clean_train_loader)} 批次, 验证: {len(self.clean_val_loader)} 批次, 测试: {len(self.clean_test_loader)} 批次")
            logger.info(f"   🔇 噪声数据 - 训练: {len(self.noisy_student_loader)} 批次, 验证: {len(self.noisy_val_loader)} 批次, 测试: {len(self.noisy_test_loader)} 批次")
            
        except Exception as e:
            logger.error(f"❌ 数据加载器设置失败: {e}")
            raise
    
    def _setup_model(self):
        """设置模型"""
        logger.info("🏗️ 设置CASIA跨域SSRL模型...")
        self.model = SSRLModel(cfg=cfg).to(self.device)
        if cfg.PRINT_MODEL_INFO:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"📊 模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
        logger.info("✅ CASIA跨域SSRL模型设置完成")

    def _load_pretrained_weights(self):
        """加载预训练权重"""
        logger.info(f"💾 加载CASIA预训练权重: {cfg.PRETRAINED_CASIA_WEIGHT}")
        if not os.path.exists(cfg.PRETRAINED_CASIA_WEIGHT):
            logger.error(f"❌ 预训练权重文件不存在!")
            raise FileNotFoundError(f"预训练权重文件不存在: {cfg.PRETRAINED_CASIA_WEIGHT}")
        
        try:
            self.model.load_complete_pretrained_weights(cfg.PRETRAINED_CASIA_WEIGHT)
            self.model._init_teacher_network()
            logger.info("✅ 成功将预训练权重加载到学生网络并同步到教师网络。")
            
        except Exception as e:
            logger.error(f"❌ 加载预训练权重失败: {e}")
            raise

    def _run_anchor_calibration(self):
        """[新增] 执行一次性的锚点校准 (论文 3.2 节) [cite: 5, 12]"""
        logger.info("⚓️ 正在执行一次性锚点校准 (Anchor Calibration)...")
        # 临时数据加载器，遍历所有数据
        clean_calib_loader, _, _, _, _ = create_casia_clean_dataloaders(cfg.CLEAN_FEAT_PATH, cfg.BATCH_SIZE * 2, fold=self.fold)
        _, _, noisy_calib_loader, _ = create_casia_noisy_dataloaders_with_speaker_isolation(cfg.NOISY_FEAT_PATH, cfg.BATCH_SIZE * 2, fold=self.fold)

        self.model.eval()
        scores_per_class = {'clean': [[] for _ in range(self.num_classes)], 'noisy': [[] for _ in range(self.num_classes)]}

        with torch.no_grad():
            # 收集干净数据分数
            for batch in clean_calib_loader:
                feats, mask, labels = batch['net_input']['feats'].to(self.device), batch['net_input']['padding_mask'].to(self.device), batch['labels'].to(self.device)
                probs = F.softmax(self.model.predict(feats, mask), dim=1)
                # [cite_start]scores, _ = DACPManager.calculate_certainty_scores(probs) # [cite: 17]
                scores, _ = DACPManager.calculate_certainty_scores(probs)
                for i, label in enumerate(labels):
                    scores_per_class['clean'][label.item()].append(scores[i].item())
            
            # 收集噪声数据分数
            for batch in noisy_calib_loader:
                feats, mask, labels = batch['net_input']['feats'].to(self.device), batch['net_input']['padding_mask'].to(self.device), batch['labels'].to(self.device)
                probs = F.softmax(self.model.predict(feats, mask), dim=1)
                # [cite_start]scores, _ = DACPManager.calculate_certainty_scores(probs) # [cite: 17]
                scores, _ = DACPManager.calculate_certainty_scores(probs)
                for i, label in enumerate(labels):
                    scores_per_class['noisy'][label.item()].append(scores[i].item())
        
        # 计算偏移和最终锚点 (公式 1-6)
        avg_scores_clean = torch.tensor([np.mean(s) if s else 0 for s in scores_per_class['clean']], device=self.device)
        avg_scores_noisy = torch.tensor([np.mean(s) if s else 0 for s in scores_per_class['noisy']], device=self.device)
        
        # [cite_start]shift_ratio = avg_scores_noisy / (avg_scores_clean + 1e-8) # [cite: 21]
        shift_ratio = avg_scores_noisy / (avg_scores_clean + 1e-8)
        
        # [cite_start]mu_clean = avg_scores_clean # [cite: 25]
        mu_clean = avg_scores_clean
        # [cite_start]sigma_clean = torch.tensor([np.std(s) if s else 0 for s in scores_per_class['clean']], device=self.device) # [cite: 26]
        sigma_clean = torch.tensor([np.std(s) if s else 0 for s in scores_per_class['clean']], device=self.device)
        
        # [cite_start]base_anchor = torch.clamp(mu_clean - cfg.ANCHOR_STD_K * sigma_clean, min=0) # [cite: 28]
        base_anchor = torch.clamp(mu_clean - cfg.ANCHOR_STD_K * sigma_clean, min=0)
        
        # [cite_start]self.calibrated_anchors = base_anchor * shift_ratio # [cite: 31]
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
        self.kl_criterion = nn.KLDivLoss(reduction='none') # 修改为 'none' 以便手动加权
        # self.scl_criterion = SupervisedContrastiveLoss(temperature=0.1) # [删除]
        
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
            self.weight_scl, self.weight_ecda, self.current_consistency_weight = 0.0, 0.0, 0.0
            return
        
        # 阶段二: 稳定教师网络 (仅一致性损失)
        if cfg.PROGRESSIVE_TRAINING:
            # 此处progress用于一致性损失的权重 ramp-up
            progress = min(1.0, (epoch - self.WARMUP_EPOCHS) / cfg.WEIGHT_RAMP_EPOCHS)
            self.current_consistency_weight = self.initial_consistency_weight + (self.final_consistency_weight - self.initial_consistency_weight) * progress
        else:
            self.current_consistency_weight = cfg.WEIGHT_CONSISTENCY
        
        # SCL 始终禁用
        self.weight_scl = 0.0
        
        # 阶段三: 启动 ECDA (在稳定阶段之后)
        if epoch >= cfg.ECDA_START_EPOCH:
            # 此处progress用于ECDA损失的权重 ramp-up
            ecda_progress = min(1.0, (epoch - cfg.ECDA_START_EPOCH) / cfg.WEIGHT_RAMP_EPOCHS)
            self.weight_ecda = cfg.WEIGHT_ECDA * ecda_progress
        else:
            self.weight_ecda = 0.0

    def train_step(self, clean_batch, noisy_batch, epoch):
        # [cite_start]1. 干净数据监督部分 (L_CE) [cite: 122]
        clean_feats, clean_padding_mask, clean_labels = clean_batch['net_input']['feats'].to(self.device), clean_batch['net_input']['padding_mask'].to(self.device), clean_batch['labels'].to(self.device)
        student_clean_encoded = self.model.student_encoder(clean_feats, clean_padding_mask)
        supervised_ce_loss = self.ce_criterion(self.model.student_classifier(student_clean_encoded), clean_labels)
        
        # SCL损失 (如果启用)
        scl_loss = torch.tensor(0.0, device=self.device) # [修改] 直接设置为0
        
        if self.is_warmup_phase(epoch):
            total_loss = supervised_ce_loss # [修改] 移除SCL
            return {'total_loss': total_loss, 'supervised_ce_loss': supervised_ce_loss, 'scl_loss': scl_loss, 'consistency_loss': torch.tensor(0.0, device=self.device), 'ecda_loss': torch.tensor(0.0, device=self.device)}
        
        # [cite_start]2. 噪声数据蒸馏部分 (L_KD, L_ECDA) [cite: 6]
        noisy_feats, noisy_padding_mask = noisy_batch['net_input']['feats'].to(self.device), noisy_batch['net_input']['padding_mask'].to(self.device)
        noisy_weak, noisy_strong = self.augmenter.weak_augment(noisy_feats), self.augmenter.strong_augment(noisy_feats)
        
        with torch.no_grad():
            teacher_encoded = self.model.teacher_encoder(noisy_weak, noisy_padding_mask)
            teacher_probs = F.softmax(self.model.teacher_classifier(teacher_encoded), dim=1)
        
        # [cite_start][修改] 使用 DACP 替代旧的置信度筛选 [cite: 7]
        if cfg.USE_DACP:
            mask, certainty_scores, class_weights_wce = self.dacp_manager.calculate_mask(
                teacher_probs, epoch, self.calibrated_anchors
            )
        else:
            # [新增] 使用固定阈值进行筛选 (消融实验)
            certainty_scores, pseudo_labels = torch.max(teacher_probs, dim=1)
            mask = (certainty_scores >= cfg.FIXED_CONFIDENCE_THRESHOLD).float()
            class_weights_wce = torch.ones_like(mask) # 固定阈值时不使用WCE

        high_confidence_count = mask.sum().item()
        
        student_strong_encoded = self.model.student_encoder(noisy_strong, noisy_padding_mask)
        student_log_probs = F.log_softmax(self.model.student_classifier(student_strong_encoded), dim=1)
        
        consistency_loss, ecda_loss = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        if high_confidence_count > 1: # ECDA 需要至少2个样本
            # [cite_start]一致性损失 (L_KD) [cite: 124]
            kl_div = self.kl_criterion(student_log_probs, teacher_probs)
            kl_div_per_sample = kl_div.sum(dim=1)
            # 加权求和，然后除以总样本数，等效于只对掩码部分求均值
            consistency_loss = (kl_div_per_sample * mask).sum() / (mask.sum() + 1e-8)
            
            # [cite_start]ECDA 损失 [cite: 126]
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

        # [cite_start]3. 计算总损失 (公式 32) [cite: 129]
        total_loss = (
            supervised_ce_loss + 
            # self.weight_scl * scl_loss + # [删除]
            self.current_consistency_weight * consistency_loss +
            self.weight_ecda * ecda_loss
        )
        
        return {
            'total_loss': total_loss, 'supervised_ce_loss': supervised_ce_loss, 
            'scl_loss': scl_loss, 'consistency_loss': consistency_loss, 'ecda_loss': ecda_loss
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
            
            # [cite_start]EMA 更新教师网络 [cite: 8]
            if not self.is_warmup_phase(epoch):
                self.model.update_teacher_ema()
                
            for key, value in losses.items():
                if isinstance(value, torch.Tensor): total_losses[key] += value.item()
            num_batches += 1
        
        # [新增] epoch 结束后更新 DACP 的类别质量分数
        if not self.is_warmup_phase(epoch):
            self.dacp_manager.update_class_quality_scores_epoch(self.dacp_manager.batch_scores_per_class)
        
        if self.scheduler: self.scheduler.step()
        return {key: value / num_batches for key, value in total_losses.items()}

    def validate(self, data_loader, domain_name="unknown"):
        """验证函数 - 计算详细评估指标"""
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in data_loader:
                feats, mask, labels = batch['net_input']['feats'].to(self.device), batch['net_input']['padding_mask'].to(self.device), batch['labels'].to(self.device)
                outputs = self.model.predict(feats, mask, use_teacher=False)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        cm = confusion_matrix(all_labels, all_preds, labels=range(self.num_classes))
        
        (prec, rec, f1, sup) = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0, labels=range(self.num_classes))

        return {
            'accuracy': accuracy_score(all_labels, all_preds) * 100,
            'weighted_accuracy': balanced_accuracy_score(all_labels, all_preds) * 100,
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100,
            'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100,
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
            best_path = os.path.join(models_dir, f'casia_cross_domain_best.pth')
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
            'info': {'noise_config': self.noise_info, 'fold': self.fold + 1, 'epoch': epoch + 1, 'is_best': is_best_result},
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
        logger.info(f"🚀 开始CASIA跨域训练 (Fold {self.fold+1}/4, 噪声: {self.noise_info['display_name']})...")
        for epoch in range(cfg.EPOCHS):
            avg_losses = self.train_epoch(epoch)
            for key, value in avg_losses.items(): self.training_history[key].append(value)
            
            should_validate = (epoch + 1) % cfg.VALIDATION_INTERVAL == 0 or not self.is_warmup_phase(epoch)
            if should_validate:
                clean_results = self.validate(self.clean_val_loader, "Clean")
                noisy_results = self.validate(self.noisy_val_loader, "Noisy")
                
                is_best = noisy_results['weighted_accuracy'] > self.best_noisy_weighted_acc + cfg.MIN_DELTA
                if is_best:
                    self.best_noisy_acc, self.best_clean_acc = noisy_results['accuracy'], clean_results['accuracy']
                    self.best_noisy_weighted_f1, self.best_clean_weighted_f1 = noisy_results['f1_weighted'], clean_results['f1_weighted']
                    self.best_noisy_weighted_acc, self.best_clean_weighted_acc = noisy_results['weighted_accuracy'], clean_results['weighted_accuracy']

                self.save_checkpoint(epoch, clean_results, noisy_results, is_best)
                
                logger.info(f"Epoch {epoch+1}/{cfg.EPOCHS} | Losses: Total={avg_losses['total_loss']:.4f} CE={avg_losses['supervised_ce_loss']:.4f} KD={avg_losses['consistency_loss']:.4f} ECDA={avg_losses['ecda_loss']:.4f} | Noisy WA: {noisy_results['weighted_accuracy']:.2f}% {'🔥' if is_best else ''}")
                
                if self.check_early_stopping(noisy_results, is_best): break
        
        # [新增] 加载最佳模型并在测试集上进行最终评估
        logger.info("🧪 开始测试集最终评估...")
        self._evaluate_on_test_set()
        
        logger.info(f"🎉 CASIA跨域训练完成 (Fold {self.fold+1})!")
        return {'best_noisy_weighted_acc': self.best_noisy_weighted_acc, 'results_dir': self.results_dir}

    def _evaluate_on_test_set(self):
        """加载最佳模型并在测试集上进行最终评估"""
        best_model_path = os.path.join(self.results_dir, "models", "casia_cross_domain_best.pth")
        if not os.path.exists(best_model_path):
            logger.warning(f"⚠️ 未找到最佳模型文件: {best_model_path}。跳过测试集评估。")
            return

        logger.info(f"💾 加载最佳模型: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 评估干净域
        logger.info("🧹 评估干净域测试集...")
        clean_test_results = self.validate(self.clean_test_loader, "Clean_Test")
        logger.info(f"   📊 Clean Domain Test - Acc: {clean_test_results['accuracy']:.2f}%, W-Acc: {clean_test_results['weighted_accuracy']:.2f}%, W-F1: {clean_test_results['f1_weighted']:.2f}%")

        # 评估噪声域
        logger.info("🔇 评估噪声域测试集...")
        noisy_test_results = self.validate(self.noisy_test_loader, "Noisy_Test")
        logger.info(f"   📊 Noisy Domain Test - Acc: {noisy_test_results['accuracy']:.2f}%, W-Acc: {noisy_test_results['weighted_accuracy']:.2f}%, W-F1: {noisy_test_results['f1_weighted']:.2f}%")

        # 保存最终测试结果
        self._save_confusion_matrices(clean_test_results, noisy_test_results, cfg.EPOCHS-1, is_best_result=False)
        self._save_detailed_results(clean_test_results, noisy_test_results, cfg.EPOCHS-1, is_best_result=False)
        
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
            }
        }
        
        test_report_path = os.path.join(self.results_dir, "reports", "FINAL_test_set_results.json")
        with open(test_report_path, 'w', encoding='utf-8') as f:
            json.dump(final_test_summary, f, indent=4, ensure_ascii=False)
        
        logger.info(f"💾 最终测试集结果已保存至: {test_report_path}")
        logger.info("✅ 测试集评估完成!")
        
        return clean_test_results, noisy_test_results

def main():
    """主函数"""
    logger.info("🚀 启动CASIA跨域训练器 (DACP+ECDA 增强版)...")
    fold = 3
    logger.info(f"📝 CASIA跨域训练 - 使用Fold {fold+1}预训练权重")
    
    trainer = FixedCASIACrossDomainTrainer(fold=fold)
    results = trainer.train()
    
    logger.info(f"✅ CASIA跨域训练完成!")
    logger.info(f"📊 最终结果统计 (核心指标: 噪声域加权准确率):")
    logger.info(f"   🎯 最佳性能: {results['best_noisy_weighted_acc']:.2f}%")
    logger.info(f"   📋 详细结果已保存至: {results['results_dir']}")

if __name__ == "__main__":
    main()