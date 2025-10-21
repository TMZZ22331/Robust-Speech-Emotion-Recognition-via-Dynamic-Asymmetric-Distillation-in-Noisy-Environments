#!/usr/bin/env python3
"""
CASIA干净数据加载器
支持说话人隔离的4折交叉验证
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

def load_casia_clean_data(feature_path, label_dict):
    """
    加载CASIA干净数据集，包括特征、标签和说话人ID
    
    Args:
        feature_path: 特征文件路径前缀（不含扩展名）
        label_dict: 标签映射字典
        
    Returns:
        dict: 包含特征、标签、说话人等信息的数据集
    """
    logger.info(f"📂 加载CASIA干净数据: {feature_path}")
    
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
    
    logger.info(f"✅ CASIA干净数据集加载完成，共 {dataset['num']} 个样本")
    return dataset

def create_casia_clean_speaker_isolated_loaders(dataset, fold, batch_size, class_names):
    """
    为CASIA干净数据集创建严格按说话人隔离的数据加载器
    4折交叉验证实现
    
    Args:
        dataset: 数据集字典
        fold: 当前折数 (0-3)
        batch_size: 批次大小
        class_names: 类别名称列表
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    all_speakers = np.unique(dataset['speakers'])
    if len(all_speakers) != 4:
        raise ValueError(f"预期CASIA有4个说话人，但找到了 {len(all_speakers)} 个")
    
    # 确定当前折的说话人分配
    test_speaker = all_speakers[fold]
    val_speaker = all_speakers[(fold + 1) % 4]  # 下一个说话人做验证
    train_speakers = [spk for spk in all_speakers if spk not in [test_speaker, val_speaker]]
    
    logger.info("📊 CASIA干净数据说话人隔离策略 (4折交叉验证):")
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
    
    def create_subset(indices, subset_name):
        """从索引列表创建数据集子集"""
        sub_labels = dataset['labels'][indices]
        sub_sizes = dataset['sizes'][indices]
        sub_offsets = dataset['offsets'][indices]
        
        # 统计子集标签分布
        label_counts = np.bincount(sub_labels, minlength=len(class_names))
        logger.info(f"   {subset_name}标签分布: {dict(zip(class_names, label_counts))}")
        
        # 创建连续的特征数组
        new_feats_list = [dataset['feats'][offset:offset+size] 
                         for offset, size in zip(sub_offsets, sub_sizes)]
        
        new_feats = np.concatenate(new_feats_list, axis=0)
        new_sizes = np.array([feat.shape[0] for feat in new_feats_list])
        new_offsets = np.concatenate([np.array([0]), np.cumsum(new_sizes)[:-1]], dtype=np.int64)
        
        return CleanEmotionDatasetFromArrays(
            new_feats, new_sizes, new_offsets, sub_labels
        )
    
    # 创建数据集
    train_dataset = create_subset(train_indices, "训练集")
    val_dataset = create_subset(val_indices, "验证集")
    test_dataset = create_subset(test_indices, "测试集")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=train_dataset.collator,
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
    
    return train_loader, val_loader, test_loader

def create_casia_clean_dataloaders(data_path, batch_size, fold=0):
    """
    创建CASIA干净数据的数据加载器
    
    Args:
        data_path: 数据路径
        batch_size: 批次大小
        fold: 当前折数
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes, class_names)
    """
    # CASIA标签映射
    label_dict = {
        'angry': 0,
        'happy': 1,
        'neutral': 2,
        'sad': 3
    }
    
    class_names = ['angry', 'happy', 'neutral', 'sad']
    
    logger.info(f"📂 创建CASIA干净数据加载器 (Fold {fold+1})...")
    
    # 加载数据
    dataset = load_casia_clean_data(data_path, label_dict)
    
    # 创建说话人隔离的数据加载器
    train_loader, val_loader, test_loader = create_casia_clean_speaker_isolated_loaders(
        dataset, fold, batch_size, class_names
    )
    
    num_classes = len(label_dict)
    
    logger.info(f"✅ CASIA干净数据加载器创建完成 (Fold {fold+1}):")
    logger.info(f"   训练批次数: {len(train_loader)} (来自 {len(train_loader.dataset)} 个样本)")
    logger.info(f"   验证批次数: {len(val_loader)} (来自 {len(val_loader.dataset)} 个样本)")
    logger.info(f"   测试批次数: {len(test_loader)} (来自 {len(test_loader.dataset)} 个样本)")
    logger.info(f"   批次大小: {batch_size}")
    logger.info(f"   类别数: {num_classes}")
    
    return train_loader, val_loader, test_loader, num_classes, class_names

if __name__ == "__main__":
    # 测试数据加载器
    import config_casia as cfg
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 测试数据加载
    train_loader, val_loader, num_classes, class_names = create_casia_clean_dataloaders(
        cfg.CLEAN_FEAT_PATH, cfg.BATCH_SIZE, fold=0
    )
    
    print(f"测试完成：类别数={num_classes}, 类别名={class_names}")
    
    # 测试一个批次
    for batch in train_loader:
        print(f"批次形状: {batch['net_input']['feats'].shape}")
        print(f"标签形状: {batch['labels'].shape}")
        break 