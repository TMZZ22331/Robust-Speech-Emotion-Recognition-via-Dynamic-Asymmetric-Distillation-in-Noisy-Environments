#!/usr/bin/env python3
"""
CASIA噪声数据加载器
支持说话人隔离的4折交叉验证
训练集不传入标签（半监督学习），验证集传入标签
"""

import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 导入基础Dataset类
from dataload_clean import CleanEmotionDatasetFromArrays

logger = logging.getLogger(__name__)

class NoisyEmotionDatasetFromArrays(Dataset):
    """
    噪声数据集类，支持无标签训练和有标签验证
    """
    
    def __init__(self, feats, sizes, offsets, labels=None, class_names=None, has_labels=True):
        """
        Args:
            feats: 特征数组
            sizes: 每个样本的长度
            offsets: 偏移量数组
            labels: 标签数组（可选，训练时为None）
            class_names: 类别名称列表
            has_labels: 是否包含标签
        """
        self.feats = feats
        self.sizes = sizes
        self.offsets = offsets
        self.labels = labels
        self.class_names = class_names or []
        self.has_labels = has_labels
        
        if has_labels and labels is None:
            raise ValueError("has_labels=True但未提供标签")
        
        self.num_samples = len(sizes)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """获取单个样本"""
        offset = self.offsets[idx]
        size = self.sizes[idx]
        
        # 提取特征
        feat = self.feats[offset:offset + size]
        
        # 构建样本字典
        sample = {
            'net_input': {
                'feats': torch.FloatTensor(feat),
                'padding_mask': torch.BoolTensor([False] * size)  # 无填充
            }
        }
        
        # 如果有标签，添加标签
        if self.has_labels and self.labels is not None:
            sample['labels'] = torch.LongTensor([self.labels[idx]])
        
        return sample
    
    def collator(self, samples):
        """批次整理函数"""
        if len(samples) == 0:
            return {}
        
        # 提取特征和填充掩码
        feats = [s['net_input']['feats'] for s in samples]
        padding_masks = [s['net_input']['padding_mask'] for s in samples]
        
        # 计算最大长度
        max_len = max(feat.size(0) for feat in feats)
        feat_dim = feats[0].size(1)
        
        # 创建批次张量
        batch_feats = torch.zeros(len(samples), max_len, feat_dim)
        batch_padding_masks = torch.ones(len(samples), max_len, dtype=torch.bool)
        
        # 填充特征和掩码
        for i, (feat, mask) in enumerate(zip(feats, padding_masks)):
            feat_len = feat.size(0)
            batch_feats[i, :feat_len] = feat
            batch_padding_masks[i, :feat_len] = mask
        
        batch = {
            'net_input': {
                'feats': batch_feats,
                'padding_mask': batch_padding_masks
            }
        }
        
        # 如果有标签，添加标签
        if self.has_labels and 'labels' in samples[0]:
            labels = [s['labels'] for s in samples]
            batch['labels'] = torch.cat(labels)
        
        return batch

def load_casia_noisy_data(feature_path, label_dict):
    """
    加载CASIA噪声数据集，包括特征、标签和说话人ID
    
    Args:
        feature_path: 特征文件路径前缀（不含扩展名）
        label_dict: 标签映射字典
        
    Returns:
        dict: 包含特征、标签、说话人等信息的数据集
    """
    logger.info(f"📂 加载CASIA噪声数据: {feature_path}")
    
    # 加载特征
    feats = np.load(f"{feature_path}.npy")
    logger.info(f"   特征形状: {feats.shape}")
    
    # 加载每个样本的长度
    with open(f"{feature_path}.lengths", "r") as f:
        sizes = [int(line.strip()) for line in f]
    logger.info(f"   样本数量: {len(sizes)}")
    
    # 加载标签并转换为数字
    with open(f"{feature_path}.lbl", "r", encoding="utf-8") as f:
        labels = [label_dict[line.strip()] for line in f]
    logger.info(f"   标签分布: {np.bincount(labels)}")
    
    # 加载说话人ID
    with open(f"{feature_path}.spk", "r", encoding="utf-8") as f:
        speakers = [line.strip() for line in f]
    
    # 统计说话人信息
    unique_speakers = np.unique(speakers)
    logger.info(f"   说话人数量: {len(unique_speakers)}")
    logger.info(f"   说话人ID: {unique_speakers}")
    
    # 计算偏移量
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    
    dataset = {
        "feats": feats,
        "sizes": np.array(sizes),
        "offsets": np.array(offsets),
        "labels": np.array(labels),
        "speakers": np.array(speakers),
        "num": len(labels),
    }
    
    logger.info(f"✅ CASIA噪声数据集加载完成，共 {dataset['num']} 个样本")
    return dataset

def create_casia_noisy_speaker_isolated_loaders(dataset, fold, batch_size, class_names):
    """
    为CASIA噪声数据集创建严格按说话人隔离的数据加载器
    4折交叉验证实现
    训练集不传入标签，验证集和测试集传入标签
    
    Args:
        dataset: 数据集字典
        fold: 当前折数 (0-3)
        batch_size: 批次大小
        class_names: 类别名称列表
        
    Returns:
        tuple: (student_loader, teacher_loader, val_loader, test_loader)
    """
    all_speakers = np.unique(dataset['speakers'])
    if len(all_speakers) != 4:
        raise ValueError(f"预期CASIA有4个说话人，但找到了 {len(all_speakers)} 个")
    
    # 确定当前折的说话人分配
    test_speaker = all_speakers[fold]
    val_speaker = all_speakers[(fold + 1) % 4]  # 下一个说话人做验证
    train_speakers = [spk for spk in all_speakers if spk not in [test_speaker, val_speaker]]
    
    logger.info("📊 CASIA噪声数据说话人隔离策略 (4折交叉验证):")
    logger.info(f"   当前折: {fold + 1}/4")
    logger.info(f"   🏋️  训练说话人: {train_speakers}")
    logger.info(f"   🧐 验证说话人: {val_speaker}")
    logger.info(f"   🧪 测试说话人: {test_speaker}")
    
    # 根据说话人ID获取样本索引
    train_indices = np.where(np.isin(dataset['speakers'], train_speakers))[0]
    val_indices = np.where(dataset['speakers'] == val_speaker)[0]
    test_indices = np.where(dataset['speakers'] == test_speaker)[0]
    
    # 打乱训练集
    np.random.shuffle(train_indices)
    
    logger.info(f"   训练样本数: {len(train_indices)}")
    logger.info(f"   验证样本数: {len(val_indices)}")
    logger.info(f"   测试样本数: {len(test_indices)}")
    
    def create_subset(indices, subset_name, has_labels=True):
        """从索引列表创建数据集子集"""
        sub_labels = dataset['labels'][indices] if has_labels else None
        sub_sizes = dataset['sizes'][indices]
        sub_offsets = dataset['offsets'][indices]
        
        # 统计子集标签分布（如果有标签）
        if has_labels:
            label_counts = np.bincount(sub_labels, minlength=len(class_names))
            logger.info(f"   {subset_name}标签分布: {dict(zip(class_names, label_counts))}")
        else:
            logger.info(f"   {subset_name}无标签训练")
        
        # 创建连续的特征数组
        new_feats_list = [dataset['feats'][offset:offset+size] 
                         for offset, size in zip(sub_offsets, sub_sizes)]
        
        new_feats = np.concatenate(new_feats_list, axis=0)
        new_sizes = np.array([feat.shape[0] for feat in new_feats_list])
        new_offsets = np.concatenate([np.array([0]), np.cumsum(new_sizes)[:-1]], dtype=np.int64)
        
        return NoisyEmotionDatasetFromArrays(
            new_feats, new_sizes, new_offsets, sub_labels, 
            class_names, has_labels
        )
    
    # 创建数据集
    # 训练集：无标签（半监督学习）
    train_dataset_student = create_subset(train_indices, "学生训练集", has_labels=False)
    train_dataset_teacher = create_subset(train_indices, "教师训练集", has_labels=False)
    
    # 验证集：有标签
    val_dataset = create_subset(val_indices, "验证集", has_labels=True)
    
    # 测试集：有标签
    test_dataset = create_subset(test_indices, "测试集", has_labels=True)
    
    # 创建数据加载器
    student_loader = DataLoader(
        train_dataset_student, 
        batch_size=batch_size, 
        collate_fn=train_dataset_student.collator,
        num_workers=0, 
        pin_memory=False, 
        shuffle=True
    )
    
    teacher_loader = DataLoader(
        train_dataset_teacher, 
        batch_size=batch_size, 
        collate_fn=train_dataset_teacher.collator,
        num_workers=0, 
        pin_memory=False, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        collate_fn=val_dataset.collator,
        num_workers=0, 
        pin_memory=False, 
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        collate_fn=test_dataset.collator,
        num_workers=0, 
        pin_memory=False, 
        shuffle=False
    )
    
    logger.info("✅ CASIA噪声数据加载器创建完成:")
    logger.info(f"   🏋️  训练样本 (Student/Teacher): {len(train_dataset_student)}")
    logger.info(f"   🧐 验证样本: {len(val_dataset)}")
    logger.info(f"   🧪 测试样本: {len(test_dataset)}")
    
    return student_loader, teacher_loader, val_loader, test_loader

def create_casia_noisy_dataloaders_with_speaker_isolation(data_path, batch_size, fold=0):
    """
    创建CASIA噪声数据的数据加载器，支持说话人隔离
    
    Args:
        data_path: 数据路径
        batch_size: 批次大小
        fold: 当前折数 (0-3)
        
    Returns:
        tuple: (student_loader, teacher_loader, val_loader, test_loader)
    """
    # CASIA标签映射
    label_dict = {
        'angry': 0,
        'happy': 1,
        'neutral': 2,
        'sad': 3
    }
    
    class_names = ['angry', 'happy', 'neutral', 'sad']
    
    logger.info(f"📂 创建CASIA噪声数据加载器 (Fold {fold+1})...")
    
    # 加载数据集
    dataset = load_casia_noisy_data(data_path, label_dict)
    
    # 创建说话人隔离的数据加载器
    student_loader, teacher_loader, val_loader, test_loader = create_casia_noisy_speaker_isolated_loaders(
        dataset, fold, batch_size, class_names
    )
    
    return student_loader, teacher_loader, val_loader, test_loader

if __name__ == "__main__":
    # 测试数据加载器
    import config_casia as cfg
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 测试数据加载
    student_loader, teacher_loader, val_loader = create_casia_noisy_dataloaders_with_speaker_isolation(
        cfg.NOISY_FEAT_PATH, cfg.BATCH_SIZE, fold=0
    )
    
    print("测试完成")
    
    # 测试学生训练批次（无标签）
    print("学生训练批次测试:")
    for batch in student_loader:
        print(f"  批次形状: {batch['net_input']['feats'].shape}")
        print(f"  有标签: {'labels' in batch}")
        break
    
    # 测试验证批次（有标签）
    print("验证批次测试:")
    for batch in val_loader:
        print(f"  批次形状: {batch['net_input']['feats'].shape}")
        print(f"  标签形状: {batch['labels'].shape}")
        break 