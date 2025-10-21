import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
# 移除'import config'，解除对特定文件的依赖
# import config as cfg

# 从您现有的 dataload_clean.py 中复用基础的Dataset类
from dataload_clean import CleanEmotionDatasetFromArrays

logger = logging.getLogger(__name__)

# load_casia_data 和 create_casia_speaker_isolated_loaders 函数保持不变...
def load_casia_data(feature_path, label_dict):
    """
    加载完整的CASIA数据集，包括特征、标签和说话人ID。
    """
    # 加载特征
    feats = np.load(f"{feature_path}.npy")
    
    # 加载每个样本的长度
    with open(f"{feature_path}.lengths", "r") as f:
        sizes = [int(line.strip()) for line in f]
    
    # 加载标签并转换为数字
    with open(f"{feature_path}.lbl", "r", encoding="utf-8") as f:
        labels = [label_dict[line.strip()] for line in f]
        
    # 加载说话人ID
    with open(f"{feature_path}.spk", "r", encoding="utf-8") as f:
        speakers = [line.strip() for line in f]
        
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
    
    print(f"✅ CASIA数据集加载完成，共 {dataset['num']} 个样本。")
    return dataset


def create_casia_speaker_isolated_loaders(dataset, fold, batch_size):
    """
    【核心函数】为CASIA数据集创建严格按说话人隔离的数据加载器。
    这是一个4折交叉验证的实现。
    - 训练集: 2个说话人
    - 验证集: 1个说话人
    - 测试集: 1个说话人 (fold对应的说话人)
    """
    all_speakers = np.unique(dataset['speakers'])
    if len(all_speakers) != 4:
        raise ValueError(f"预期CASIA有4个说话人, 但在.spk文件中找到了 {len(all_speakers)} 个。")

    # 确定当前折的说话人分配
    test_speaker = all_speakers[fold]
    val_speaker = all_speakers[(fold + 1) % 4] # 使用下一个说话人做验证
    train_speakers = [spk for spk in all_speakers if spk not in [test_speaker, val_speaker]]

    print("📊 [CASIA Data] Speaker Isolation Strategy (4-Fold):")
    print(f"  -  Fold: {fold + 1}/4")
    print(f"  - 🏋️  Training Speakers: {train_speakers}")
    print(f"  - 🧐 Validation Speaker: {val_speaker}")
    print(f"  - 🧪 Test Speaker: {test_speaker}")

    # 根据说话人ID获取样本索引
    train_indices = np.where(np.isin(dataset['speakers'], train_speakers))[0]
    val_indices = np.where(dataset['speakers'] == val_speaker)[0]
    test_indices = np.where(dataset['speakers'] == test_speaker)[0]

    np.random.shuffle(train_indices) # 打乱训练集

    def create_subset(indices):
        """辅助函数：从索引列表创建数据集子集"""
        sub_labels = dataset['labels'][indices]
        sub_sizes = dataset['sizes'][indices]
        sub_offsets = dataset['offsets'][indices]
        
        # 为了创建新的连续特征数组，我们需要从原始feats中提取片段
        # 注意：这部分会消耗更多内存，但能确保数据隔离和Dataset类的兼容性
        new_feats_list = [dataset['feats'][offset:offset+size] for offset, size in zip(sub_offsets, sub_sizes)]
        
        new_feats = np.concatenate(new_feats_list, axis=0)
        new_sizes = np.array([feat.shape[0] for feat in new_feats_list])
        new_offsets = np.concatenate([np.array([0]), np.cumsum(new_sizes)[:-1]], dtype=np.int64)

        return CleanEmotionDatasetFromArrays(new_feats, new_sizes, new_offsets, sub_labels, class_names=['angry', 'happy', 'neutral', 'sad'])

    # 创建数据集
    train_dataset = create_subset(train_indices)
    val_dataset = create_subset(val_indices)
    test_dataset = create_subset(test_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collator, num_workers=0, pin_memory=False, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_dataset.collator, num_workers=0, pin_memory=False, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collator, num_workers=0, pin_memory=False, shuffle=False)
                            
    return train_loader, val_loader, test_loader

# --- 重点修改区域 ---
def create_casia_dataloaders(config, fold=0):
    """
    【对外接口】为CASIA数据集创建加载器。
    这个函数现在直接接收一个config对象。
    """
    batch_size = config.BATCH_SIZE
    data_path = config.FEAT_PATH
    label_dict = config.LABEL_DICT
    
    print("📂 正在加载CASIA数据 (严格说话人隔离模式)...")
    
    dataset = load_casia_data(data_path, label_dict)
    
    train_loader, val_loader, test_loader = create_casia_speaker_isolated_loaders(
        dataset, fold, batch_size
    )
    
    num_classes = len(label_dict)
    idx2label = {v: k for k, v in label_dict.items()}
    
    print(f"✅ CASIA数据加载器创建完成 (Fold {fold+1}):")
    print(f"   - 训练批次数: {len(train_loader)} (来自 {len(train_loader.dataset)} 个样本)")
    print(f"   - 验证批次数: {len(val_loader)} (来自 {len(val_loader.dataset)} 个样本)")
    print(f"   - 测试批次数: {len(test_loader)} (来自 {len(test_loader.dataset)} 个样本)")
    print(f"   - 批次大小: {batch_size}")
    
    return train_loader, val_loader, test_loader, num_classes, idx2label 