import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import logging
from collections import defaultdict, deque
from scipy.stats import mode
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import config_emodb as cfg

# 设置日志
logger = logging.getLogger(__name__)

class Emotion2VecUtils:
    """
    Emotion2Vec相关的工具类
    """
    
    @staticmethod
    def load_emotion2vec_features(data_path):
        """
        加载emotion2vec预处理后的特征
        
        Args:
            data_path: 数据路径前缀（不包含扩展名）
            
        Returns:
            dict: 包含特征数据的字典
        """
        try:
            # 加载特征数据
            npy_data = np.load(data_path + ".npy")
            
            # 加载长度信息
            with open(data_path + ".lengths", "r") as f:
                lengths = [int(line.strip()) for line in f.readlines()]
            
            # 尝试加载标签
            labels = None
            label_file = data_path + ".emo"
            if os.path.exists(label_file):
                with open(label_file, "r") as f:
                    labels = [line.strip().split()[1] for line in f.readlines()]
            
            logger.info(f"✅ 成功加载emotion2vec特征: {data_path}")
            logger.info(f"   特征形状: {npy_data.shape}")
            logger.info(f"   样本数量: {len(lengths)}")
            logger.info(f"   标签数量: {len(labels) if labels else 0}")
            
            return {
                'features': npy_data,
                'lengths': lengths,
                'labels': labels,
                'feature_dim': npy_data.shape[1]
            }
            
        except Exception as e:
            logger.error(f"❌ 加载emotion2vec特征失败: {data_path}, 错误: {e}")
            raise
    
    @staticmethod
    def compute_padding_mask(batch_features, max_length=None):
        """
        计算批次的padding mask
        
        Args:
            batch_features: 特征列表，每个元素为 (seq_len, feature_dim)
            max_length: 最大长度，如果为None则使用批次中的最大长度
            
        Returns:
            tuple: (padded_features, padding_mask)
        """
        if max_length is None:
            max_length = max(feat.shape[0] for feat in batch_features)
        
        batch_size = len(batch_features)
        feature_dim = batch_features[0].shape[1]
        
        # 创建填充后的特征张量
        padded_features = torch.zeros(batch_size, max_length, feature_dim)
        padding_mask = torch.ones(batch_size, max_length, dtype=torch.bool)
        
        for i, feat in enumerate(batch_features):
            seq_len = feat.shape[0]
            padded_features[i, :seq_len] = feat
            padding_mask[i, :seq_len] = False
        
        return padded_features, padding_mask
    
    @staticmethod
    def validate_emotion2vec_data(data_path):
        """
        验证emotion2vec数据的完整性
        
        Args:
            data_path: 数据路径前缀
            
        Returns:
            bool: 数据是否有效
        """
        try:
            required_files = [
                data_path + ".npy",
                data_path + ".lengths"
            ]
            
            for file_path in required_files:
                if not os.path.exists(file_path):
                    logger.error(f"❌ 缺少必需文件: {file_path}")
                    return False
            
            # 验证数据一致性
            npy_data = np.load(data_path + ".npy")
            with open(data_path + ".lengths", "r") as f:
                lengths = [int(line.strip()) for line in f.readlines()]
            
            total_length = sum(lengths)
            if total_length != npy_data.shape[0]:
                logger.error(f"❌ 长度不匹配: 总长度={total_length}, 特征行数={npy_data.shape[0]}")
                return False
            
            logger.info(f"✅ emotion2vec数据验证通过: {data_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据验证失败: {e}")
            return False

class ModelUtils:
    """
    模型相关的工具函数
    """
    
    @staticmethod
    def count_parameters(model):
        """
        统计模型参数数量
        
        Args:
            model: PyTorch模型
            
        Returns:
            dict: 参数统计信息
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0
        }
    
    @staticmethod
    def print_model_info(model, model_name="Model"):
        """
        打印模型信息
        
        Args:
            model: PyTorch模型
            model_name: 模型名称
        """
        param_info = ModelUtils.count_parameters(model)
        
        logger.info(f"📊 {model_name} 参数统计:")
        logger.info(f"   总参数: {param_info['total_params']:,}")
        logger.info(f"   可训练参数: {param_info['trainable_params']:,}")
        logger.info(f"   冻结参数: {param_info['frozen_params']:,}")
        logger.info(f"   可训练比例: {param_info['trainable_ratio']:.2%}")
    
    @staticmethod
    def save_model_checkpoint(model, optimizer, epoch, metrics, save_path):
        """
        保存模型检查点
        
        Args:
            model: PyTorch模型
            optimizer: 优化器
            epoch: 当前epoch
            metrics: 评估指标字典
            save_path: 保存路径
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'model_info': ModelUtils.count_parameters(model)
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"💾 模型检查点已保存: {save_path}")
    
    @staticmethod
    def load_model_checkpoint(model, optimizer, checkpoint_path):
        """
        加载模型检查点
        
        Args:
            model: PyTorch模型
            optimizer: 优化器
            checkpoint_path: 检查点路径
            
        Returns:
            dict: 检查点信息
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"📥 模型检查点已加载: {checkpoint_path}")
        logger.info(f"   Epoch: {checkpoint['epoch']}")
        logger.info(f"   指标: {checkpoint.get('metrics', {})}")
        
        return checkpoint

class MetricsCalculator:
    """
    评估指标计算器
    """
    
    @staticmethod
    def compute_classification_metrics(y_true, y_pred, class_names=None, average='weighted'):
        """
        计算分类任务的各种指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称列表
            average: 平均方式
            
        Returns:
            dict: 包含各种指标的字典
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average=average),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0)
        }
        
        # 计算每个类别的指标
        if class_names is not None:
            per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
            for i, class_name in enumerate(class_names):
                if i < len(per_class_f1):
                    metrics[f'f1_{class_name}'] = per_class_f1[i]
        
        return metrics
    
    @staticmethod
    def compute_confidence_stats(logits, threshold=0.8):
        """
        计算置信度统计信息
        
        Args:
            logits: 模型输出的logits
            threshold: 置信度阈值
            
        Returns:
            dict: 置信度统计信息
        """
        probs = F.softmax(logits, dim=1)
        max_probs, pred_labels = torch.max(probs, dim=1)
        
        high_confidence_mask = max_probs >= threshold
        high_confidence_count = high_confidence_mask.sum().item()
        total_count = logits.shape[0]
        
        return {
            'high_confidence_count': high_confidence_count,
            'total_count': total_count,
            'high_confidence_ratio': high_confidence_count / total_count if total_count > 0 else 0,
            'mean_confidence': max_probs.mean().item(),
            'std_confidence': max_probs.std().item(),
            'min_confidence': max_probs.min().item(),
            'max_confidence': max_probs.max().item()
        }
    
    @staticmethod
    def print_classification_report(y_true, y_pred, class_names=None, title="Classification Report"):
        """
        打印详细的分类报告
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称列表
            title: 报告标题
        """
        logger.info(f"\n📊 {title}")
        logger.info("=" * 60)
        
        # 基本指标
        metrics = MetricsCalculator.compute_classification_metrics(y_true, y_pred, class_names)
        logger.info(f"准确率: {metrics['accuracy']:.4f}")
        logger.info(f"F1分数: {metrics['f1_score']:.4f}")
        logger.info(f"精确率: {metrics['precision']:.4f}")
        logger.info(f"召回率: {metrics['recall']:.4f}")
        
        # 详细报告
        if class_names is not None:
            target_names = class_names
        else:
            target_names = None
            
        report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
        logger.info(f"\n详细报告:\n{report}")

class DataAugmentation:
    """
    统一的数据增强模块：教师弱增强 vs 学生强增强
    支持完整的增强策略：噪声注入、特征dropout、时序遮盖
    """
    def __init__(self, weak_noise_std=None, strong_noise_std=None, dropout_rate=None, temporal_mask_ratio=None):
        self.weak_noise_std = weak_noise_std if weak_noise_std is not None else cfg.WEAK_NOISE_STD
        self.strong_noise_std = strong_noise_std if strong_noise_std is not None else cfg.STRONG_NOISE_STD
        self.dropout_rate = dropout_rate if dropout_rate is not None else cfg.DROPOUT_RATE
        self.temporal_mask_ratio = temporal_mask_ratio if temporal_mask_ratio is not None else cfg.TEMPORAL_MASK_RATIO
    
    def weak_augment(self, data):
        """教师网络弱增强：仅添加小幅高斯噪声"""
        noise = torch.randn_like(data) * self.weak_noise_std
        return data + noise
    
    def strong_augment(self, data):
        """学生网络强增强：噪声注入 + 特征dropout + 时序遮盖"""
        augmented = data.clone()
        
        # 1. 噪声注入
        noise = torch.randn_like(augmented) * self.strong_noise_std
        augmented = augmented + noise
        
        # 2. 特征级dropout
        if self.dropout_rate > 0:
            feature_mask = torch.rand(augmented.shape[-1], device=augmented.device) > self.dropout_rate
            augmented = augmented * feature_mask.float()
        
        # 3. 时序遮盖（如果是序列数据）
        if self.temporal_mask_ratio > 0 and len(augmented.shape) >= 2:
            augmented = self._apply_temporal_masking(augmented)
        
        return augmented
    
    def _apply_temporal_masking(self, data):
        """应用时序遮盖"""
        if len(data.shape) == 2:  # [seq_len, feature_dim]
            seq_len = data.shape[0]
            mask_len = int(seq_len * self.temporal_mask_ratio)
            
            if mask_len > 0:
                start_idx = torch.randint(0, max(1, seq_len - mask_len + 1), (1,)).item()
                data_masked = data.clone()
                data_masked[start_idx:start_idx + mask_len] = 0
                return data_masked
                
        elif len(data.shape) == 3:  # [batch_size, seq_len, feature_dim]
            batch_size, seq_len = data.shape[0], data.shape[1]
            mask_len = int(seq_len * self.temporal_mask_ratio)
            
            if mask_len > 0:
                data_masked = data.clone()
                for i in range(batch_size):
                    start_idx = torch.randint(0, max(1, seq_len - mask_len + 1), (1,)).item()
                    data_masked[i, start_idx:start_idx + mask_len] = 0
                return data_masked
        
        return data

# ===== 以下是根据您的新方法添加的模块 =====

class DACPManager:
    """
    动态自适应置信度剪枝 (DACP) 管理器
    实现论文 3.3 节的完整逻辑
    """
    def __init__(self, num_classes, total_epochs, device):
        self.num_classes = num_classes
        self.total_epochs = total_epochs
        self.device = device
        
        # 阶段二：类别表现追踪 (公式 10)
        # 每个类别的伪标签质量分数 Q_c^e, 初始化为0.5
        self.class_quality_scores = torch.full((num_classes,), 0.5, device=self.device)

        # 阶段四：最终阈值平滑 (公式 17)
        # 每个类别的最终阈值 τ_c^t, 初始化为0.5
        self.ema_thresholds = torch.full((num_classes,), 0.5, device=self.device)
        
        # 批次中每个类别的确定性分数集合，用于计算分位数
        self.batch_scores_per_class = [[] for _ in range(num_classes)]

    @staticmethod
    def calculate_certainty_scores(probs):
        """
        阶段一：计算修正后的确定性分数 (公式 8)
        Args:
            probs: 教师网络输出的概率分布 [B, C]
        Returns:
            scores: 每个样本的确定性分数 [B]
            preds: 每个样本的预测类别 [B]
        """
        max_probs, preds = torch.max(probs, dim=1)
        
        # 计算熵 (公式 7)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-8), dim=1)
        
        # 归一化熵
        normalized_entropy = entropy / np.log2(probs.size(1))
        
        # 计算确定性分数 (公式 8)
        certainty_scores = max_probs * (1 - normalized_entropy)
        return certainty_scores, preds

    def update_class_quality_scores_epoch(self, epoch_scores_per_class):
        """
        阶段二：更新每个类别的整体质量分数 Q_c^e (公式 9, 10)
        应在每个 epoch 结束后调用
        """
        current_epoch_quality = torch.tensor(
            [np.mean(scores) if len(scores) > 0 else self.class_quality_scores[i].item() for i, scores in enumerate(epoch_scores_per_class)],
            device=self.device,
            dtype=torch.float32
        )
        
        # EMA 更新 (公式 10)
        self.class_quality_scores = (
            cfg.DACP_QUALITY_SMOOTHING_BETA * self.class_quality_scores +
            (1 - cfg.DACP_QUALITY_SMOOTHING_BETA) * current_epoch_quality
        )
        # 清空用于存储整个epoch分数的列表
        self.batch_scores_per_class = [[] for _ in range(self.num_classes)]

    def calculate_mask(self, teacher_probs, epoch, calibrated_anchors):
        """
        计算最终的置信度掩码 (DACP 核心逻辑)
        Args:
            teacher_probs: 教师网络输出的概率 [B, C]
            epoch: 当前轮次
            calibrated_anchors: 预先计算好的动态校准锚点 [C]
        Returns:
            mask: 高置信度样本的二值掩码 [B]
            certainty_scores: 所有样本的确定性分数 [B]
            class_weights_wce: 类别权重 W_c^e [C]
        """
        # 阶段一：计算确定性分数
        certainty_scores, preds = self.calculate_certainty_scores(teacher_probs)
        
        # --- 阶段三 & 四 开始 ---
        # 1. 计算相对表现差距 Δ_c^e 和类别权重 W_c^e (公式 11, 12)
        avg_quality = self.class_quality_scores.mean()
        delta_ce = self.class_quality_scores - avg_quality
        # 使用 sigmoid 计算类别权重 (公式 12)
        class_weights_wce = torch.sigmoid(cfg.DACP_SENSITIVITY_K * delta_ce)

        # 2. 计算动态筛选标准 γ_e (公式 13)
        progress = epoch / self.total_epochs
        gamma_e = cfg.DACP_QUANTILE_START + (cfg.DACP_QUANTILE_END - cfg.DACP_QUANTILE_START) * progress
        
        # 3. 计算批次内各类别基础阈值 τ_c^t (公式 14)
        batch_thresholds_tct = torch.zeros(self.num_classes, device=self.device)
        for c in range(self.num_classes):
            class_scores = certainty_scores[preds == c]
            if len(class_scores) > 0:
                # Quantile(U_c^t, γ_e)
                batch_thresholds_tct[c] = torch.quantile(class_scores, gamma_e)
            else:
                # 如果批次内没有该类别，使用EMA平滑后的历史阈值
                batch_thresholds_tct[c] = self.ema_thresholds[c]

        # 4. 计算动态调整项和融合锚点 (公式 15, 16)
        dynamic_adjustment = cfg.DACP_CALIBRATION_STRENGTH_LAMBDA * (class_weights_wce - 0.5)
        dynamic_thresholds = batch_thresholds_tct + dynamic_adjustment
        
        # 融合锚点，确保不低于底线 (公式 16)
        floored_thresholds = torch.max(dynamic_thresholds, calibrated_anchors.to(self.device))

        # 5. EMA平滑最终阈值 (公式 17)
        self.ema_thresholds = (
            cfg.DACP_THRESHOLD_SMOOTHING_ALPHA * self.ema_thresholds +
            (1 - cfg.DACP_THRESHOLD_SMOOTHING_ALPHA) * floored_thresholds
        )

        # 6. 生成最终掩码 (公式 18)
        final_thresholds_per_sample = self.ema_thresholds[preds]
        mask = certainty_scores >= final_thresholds_per_sample
        
        # 为下一个epoch的质量更新收集分数
        for c in range(self.num_classes):
            self.batch_scores_per_class[c].extend(certainty_scores[preds == c].detach().cpu().numpy())

        return mask, certainty_scores, class_weights_wce


class ECDALoss(nn.Module):
    """
    基于能量最小化的类别感知分布对齐 (ECDA) 损失
    实现论文 3.4 节的完整逻辑
    """
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def _gaussian_kernel(self, source, target, sample_weights_s, sample_weights_t):
        """
        修改版高斯核，用于计算带样本权重的 MMD
        Args:
            source: 源域特征 [N_s, D]
            target: 目标域特征 [N_t, D]
            sample_weights_s: 源域样本权重 [N_s]
            sample_weights_t: 目标域样本权重 [N_t]
        Returns:
            Term_ss, Term_tt, Term_st
        """
        n_s, n_t = source.size(0), target.size(0)
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        
        L2_dist = ((total0 - total1)**2).sum(2)
        
        # MMD original paper logic to calculate bandwidth
        bandwidth = torch.sum(L2_dist.data) / ( (n_s + n_t)**2 - (n_s + n_t) ) if (n_s + n_t) > 1 else torch.tensor(1.0)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_dist / (bw + 1e-8)) for bw in bandwidth_list]
        kernel_matrix = sum(kernel_val)

        # 提取各个块
        K_ss = kernel_matrix[:n_s, :n_s]
        K_tt = kernel_matrix[n_s:, n_s:]
        K_st = kernel_matrix[:n_s, n_s:]
        
        # 应用注意力权重 (公式 21, 22, 23)
        weights_s_matrix = torch.outer(sample_weights_s, sample_weights_s)
        weights_t_matrix = torch.outer(sample_weights_t, sample_weights_t)
        weights_st_matrix = torch.outer(sample_weights_s, sample_weights_t)

        # Term_clean (公式 21)
        term_ss = (K_ss * weights_s_matrix).sum() / (weights_s_matrix.sum() + 1e-8)
        # Term_noisy (公式 22)
        term_tt = (K_tt * weights_t_matrix).sum() / (weights_t_matrix.sum() + 1e-8)
        # Term_cross (公式 23)
        term_st = (K_st * weights_st_matrix).sum() / (weights_st_matrix.sum() + 1e-8)

        return term_ss, term_tt, term_st

    def forward(self, clean_feats, noisy_feats, clean_labels, noisy_labels, noisy_mask, noisy_scores, class_weights_wce):
        total_loss = torch.tensor(0.0, device=clean_feats.device)
        num_classes = class_weights_wce.size(0)

        # 检查noisy_mask类型，如果不是布尔类型则转换
        if noisy_mask.dtype != torch.bool:
            # 当没有使用DACP时，noisy_mask实际上是置信度分数
            # 使用固定阈值将其转换为布尔掩码
            noisy_mask = noisy_mask > cfg.FIXED_CONFIDENCE_THRESHOLD

        # 计算所有含噪特征簇的质心 (用于类间斥力)
        noisy_centroids = []
        valid_centroid_classes = []
        for c in range(num_classes):
            class_noisy_feats = noisy_feats[(noisy_labels == c) & noisy_mask]
            if len(class_noisy_feats) > 0:
                noisy_centroids.append(class_noisy_feats.mean(dim=0))
                valid_centroid_classes.append(c)
        
        # 计算类间斥力损失 (公式 27)
        repulsion_loss = torch.tensor(0.0, device=clean_feats.device)
        if len(noisy_centroids) > 1:
            centroids_tensor = torch.stack(noisy_centroids)
            dist_matrix = torch.pdist(centroids_tensor, p=2)
            repulsion_loss = -dist_matrix.mean()
        
        # 计算类别级注意力权重 (公式 24)
        avg_class_weight = class_weights_wce.mean()
        class_attention_weights = torch.exp(cfg.ECDA_CLASS_ATTENTION_LAMBDA * (avg_class_weight - class_weights_wce))
        
        for c in range(num_classes):
            # 1. 提取每个类别的特征 (公式 19)
            class_clean_feats = clean_feats[clean_labels == c]
            
            noisy_class_mask = (noisy_labels == c) & noisy_mask
            class_noisy_feats = noisy_feats[noisy_class_mask]
            
            # 计算可行性门控 (论文 3.4 阶段二)
            if len(class_clean_feats) < 2 or len(class_noisy_feats) < 2:
                continue

            # 2. 注意力加权MMD (公式 20)
            clean_weights = torch.ones(len(class_clean_feats), device=clean_feats.device)
            noisy_weights = noisy_scores[noisy_class_mask]  # 样本级注意力
            
            term_ss, term_tt, term_st = self._gaussian_kernel(class_clean_feats, class_noisy_feats, clean_weights, noisy_weights)
            mmd_attn_loss = term_ss + term_tt - 2 * term_st

            # 3. 分布紧凑性正则化 (公式 25)
            noisy_centroid = class_noisy_feats.mean(dim=0)
            compactness_loss = torch.mean(torch.sum((class_noisy_feats - noisy_centroid)**2, dim=1))
            
            # 4. 最终对齐损失 (公式 28)
            # L_repulsion 是全局的，但我们在每个类别循环中都加上它, 最终会被类别权重缩放
            ecda_loss_c = (
                mmd_attn_loss + 
                cfg.ECDA_COMPACTNESS_WEIGHT_GAMMA * compactness_loss +
                cfg.ECDA_REPULSION_WEIGHT_DELTA * repulsion_loss
            )
            
            # 5. 总损失加权求和 (公式 29)
            total_loss += class_attention_weights[c] * ecda_loss_c

        return total_loss
