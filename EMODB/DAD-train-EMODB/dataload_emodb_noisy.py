#!/usr/bin/env python3
"""
EmoDB噪声数据加载器
支持说话人隔离的5折交叉验证
10个说话人，每折2个说话人
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

# EmoDB说话人ID列表（10折交叉验证，每个说话人独立一折）
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

class NoisyEmotionDatasetFromArrays(Dataset):
    """
    EmoDB噪声数据集类，支持无标签训练和有标签验证
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

def load_emodb_noisy_data(feature_path, label_dict):
    """
    加载EmoDB噪声数据集，包括特征、标签和说话人ID
    
    Args:
        feature_path: 特征文件路径前缀（不含扩展名）
        label_dict: 标签映射字典
        
    Returns:
        dict: 包含特征、标签、说话人等信息的数据集
    """
    logger.info(f"📂 加载EmoDB噪声数据: {feature_path}")
    
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
    
    logger.info(f"✅ EmoDB噪声数据集加载完成，共 {dataset['num']} 个样本")
    return dataset

def create_emodb_noisy_speaker_isolated_loaders(dataset, fold, batch_size, class_names):
    """
    为EmoDB噪声数据集创建严格按说话人隔离的数据加载器
    10折交叉验证：8个说话人训练、1个说话人验证、1个说话人测试
    训练集不传入标签，验证集和测试集传入标签
    
    Args:
        dataset: 数据集字典
        fold: 当前折数 (0-9)
        batch_size: 批次大小
        class_names: 类别名称列表
        
    Returns:
        tuple: (student_loader, teacher_loader, val_loader, test_loader)
    """
    if fold < 0 or fold >= 10:
        raise ValueError(f"fold必须在0-9范围内，当前: {fold}")
    
    # 获取当前fold的说话人分配
    train_speakers, val_speaker, test_speaker = get_emodb_fold_speakers(fold)
    
    logger.info("📊 EmoDB噪声数据说话人隔离策略 (10折交叉验证):")
    logger.info(f"   当前折: {fold + 1}/10")
    logger.info(f"   🏋️  训练说话人: {train_speakers} (8个说话人)")
    logger.info(f"   🧐 验证说话人: {val_speaker} (1个说话人)")
    logger.info(f"   🧪 测试说话人: {test_speaker} (1个说话人)")
    
    # 根据说话人ID获取样本索引
    # 需要从'emodb_spk_XX'格式中提取数字部分进行匹配
    def extract_speaker_id(speaker_full_id):
        """从完整说话人ID提取数字部分"""
        return speaker_full_id.split('_')[-1]
    
    # 提取数据集中所有说话人的数字ID
    dataset_speaker_ids = np.array([extract_speaker_id(spk) for spk in dataset['speakers']])
    
    train_indices = np.where(np.isin(dataset_speaker_ids, train_speakers))[0]
    val_indices = np.where(dataset_speaker_ids == val_speaker)[0]
    test_indices = np.where(dataset_speaker_ids == test_speaker)[0]
    
    # 打乱训练集
    np.random.shuffle(train_indices)
    
    logger.info(f"   训练样本数: {len(train_indices)}")
    logger.info(f"   验证样本数: {len(val_indices)}")
    logger.info(f"   测试样本数: {len(test_indices)}")
    
    # 验证说话人隔离
    train_speakers_actual_full = np.unique(dataset['speakers'][train_indices])
    val_speakers_actual_full = np.unique(dataset['speakers'][val_indices])
    test_speakers_actual_full = np.unique(dataset['speakers'][test_indices])
    
    # 提取数字ID用于日志显示
    train_speakers_actual = [extract_speaker_id(spk) for spk in train_speakers_actual_full]
    val_speakers_actual = [extract_speaker_id(spk) for spk in val_speakers_actual_full]
    test_speakers_actual = [extract_speaker_id(spk) for spk in test_speakers_actual_full]
    
    logger.info(f"   实际训练说话人: {train_speakers_actual}")
    logger.info(f"   实际验证说话人: {val_speakers_actual}")
    logger.info(f"   实际测试说话人: {test_speakers_actual}")
    
    # 确保没有重叠（使用数字ID进行检查）
    assert len(set(train_speakers_actual) & set(val_speakers_actual)) == 0, "训练集和验证集说话人重叠"
    assert len(set(train_speakers_actual) & set(test_speakers_actual)) == 0, "训练集和测试集说话人重叠"
    assert len(set(val_speakers_actual) & set(test_speakers_actual)) == 0, "验证集和测试集说话人重叠"
    
    def create_subset(indices, subset_name, has_labels=True):
        """从索引列表创建数据集子集"""
        if len(indices) == 0:
            raise ValueError(f"无法创建{subset_name}：没有找到匹配的样本")
        
        sub_labels = dataset['labels'][indices] if has_labels else None
        sub_sizes = dataset['sizes'][indices]
        sub_offsets = dataset['offsets'][indices]
        
        # 统计子集标签分布（如果有标签）
        if has_labels:
            label_counts = np.bincount(sub_labels, minlength=len(class_names))
            logger.info(f"   {subset_name}标签分布: {dict(zip(class_names, label_counts))}")
        else:
            logger.info(f"   {subset_name}无标签训练（半监督学习）")
        
        # 创建连续的特征数组
        new_feats_list = [dataset['feats'][offset:offset+size] 
                         for offset, size in zip(sub_offsets, sub_sizes)]
        
        if len(new_feats_list) == 0:
            raise ValueError(f"无法创建{subset_name}：特征列表为空")
        
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
    
    logger.info("✅ EmoDB噪声数据加载器创建完成:")
    logger.info(f"   🏋️  训练样本 (Student/Teacher): {len(train_dataset_student)}")
    logger.info(f"   🧐 验证样本: {len(val_dataset)}")
    logger.info(f"   🧪 测试样本: {len(test_dataset)}")
    
    return student_loader, teacher_loader, val_loader, test_loader

def create_emodb_noisy_dataloaders_with_speaker_isolation(data_path, batch_size, fold=0):
    """
    创建EmoDB噪声数据的数据加载器，支持说话人隔离
    
    Args:
        data_path: 数据路径
        batch_size: 批次大小
        fold: 当前折数 (0-9)
        
    Returns:
        tuple: (student_loader, teacher_loader, val_loader, test_loader)
    """
    # EmoDB标签映射
    label_dict = {
        'angry': 0,
        'happy': 1,
        'neutral': 2,
        'sad': 3
    }
    
    class_names = ['angry', 'happy', 'neutral', 'sad']
    
    logger.info(f"📂 创建EmoDB噪声数据加载器 (Fold {fold+1})...")
    
    # 加载数据集
    dataset = load_emodb_noisy_data(data_path, label_dict)
    
    # 创建说话人隔离的数据加载器
    student_loader, teacher_loader, val_loader, test_loader = create_emodb_noisy_speaker_isolated_loaders(
        dataset, fold, batch_size, class_names
    )
    
    return student_loader, teacher_loader, val_loader, test_loader

if __name__ == "__main__":
    # 测试数据加载器
    import config_emodb as cfg
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 测试数据加载
    for fold in range(5):
        print(f"\n测试 Fold {fold+1}:")
        student_loader, teacher_loader, val_loader = create_emodb_noisy_dataloaders_with_speaker_isolation(
            cfg.NOISY_FEAT_PATH, cfg.BATCH_SIZE, fold=fold
        )
        
        print(f"Fold {fold+1} 测试完成")
        
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