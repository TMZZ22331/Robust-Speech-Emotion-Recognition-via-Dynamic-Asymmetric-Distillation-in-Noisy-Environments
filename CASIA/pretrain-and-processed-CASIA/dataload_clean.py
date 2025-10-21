import os
import logging
import contextlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config as cfg

logger = logging.getLogger(__name__)

# #############################################################################
# 以下代码与您提供的版本相同，无需修改
# load_emotion2vec_dataset, CleanEmotionDataset, CleanEmotionDatasetFromArrays,
# load_ssl_features
# #############################################################################

def load_emotion2vec_dataset(data_path, labels='emo', min_length=3, max_length=None):
    """
    加载emotion2vec格式的数据集
    """
    sizes = []
    offsets = []
    emo_labels = []
    npy_data = np.load(data_path + ".npy")
    offset = 0
    skipped = 0
    label_file_path = data_path + f".{labels}"
    if not os.path.exists(label_file_path):
        labels = None
        logger.warning(f"标签文件不存在: {label_file_path}")
    with open(data_path + ".lengths", "r") as len_f, open(
        label_file_path, "r"
    ) if labels is not None else contextlib.ExitStack() as lbl_f:
        for line in len_f:
            length = int(line.rstrip())
            lbl = None if labels is None else next(lbl_f).rstrip().split()[1]
            if length >= min_length and (max_length is None or length <= max_length):
                sizes.append(length)
                offsets.append(offset)
                if lbl is not None:
                    emo_labels.append(lbl)
            else:
                skipped += 1
            offset += length
    sizes = np.asarray(sizes)
    offsets = np.asarray(offsets)
    logger.info(f"加载了 {len(offsets)} 个样本，跳过了 {skipped} 个样本")
    return npy_data, sizes, offsets, emo_labels

class CleanEmotionDataset(Dataset):
    def __init__(self, data_path, transform=None, min_length=3, max_length=None):
        self.transform = transform
        self.feats, self.sizes, self.offsets, self.labels = load_emotion2vec_dataset(
            data_path, labels='emo', min_length=min_length, max_length=max_length
        )
        if self.labels:
            self.numerical_labels = [cfg.LABEL_DICT[label] for label in self.labels]
        else:
            self.numerical_labels = None
        self.class_names = list(cfg.LABEL_DICT.keys())
        self.num_classes = len(self.class_names)
        print(f"✅ 干净数据集加载完成:")
        print(f"   📊 样本数量: {len(self.sizes)}")
        print(f"   📏 特征维度: {self.feats.shape[1]}")
        print(f"   🏷️ 类别数量: {self.num_classes}")
        print(f"   📋 类别标签: {self.class_names}")
        if self.labels:
            from collections import Counter
            label_counts = Counter(self.labels)
            for label, count in label_counts.items():
                print(f"      {label}: {count} 样本")
    def __len__(self):
        return len(self.sizes)
    def __getitem__(self, index):
        offset = self.offsets[index]
        size = self.sizes[index]
        end = offset + size
        feats = torch.from_numpy(self.feats[offset:end, :].copy()).float()
        res = {"id": index, "feats": feats}
        if self.numerical_labels is not None:
            res["target"] = self.numerical_labels[index]
        else:
            res["target"] = None
        return res
    def collator(self, samples):
        if len(samples) == 0: return {}
        feats = [s["feats"] for s in samples]
        sizes = [s.shape[0] for s in feats]
        labels = None
        if samples[0]["target"] is not None:
            labels = torch.tensor([s["target"] for s in samples], dtype=torch.long)
        target_size = max(sizes)
        collated_feats = feats[0].new_zeros(len(feats), target_size, feats[0].size(-1))
        padding_mask = torch.BoolTensor(len(feats), target_size).fill_(False)
        for i, (feat, size) in enumerate(zip(feats, sizes)):
            collated_feats[i, :size] = feat
            padding_mask[i, size:] = True
        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {"feats": collated_feats, "padding_mask": padding_mask},
            "labels": labels
        }
        return res
    def get_class_names(self): return self.class_names
    def get_num_classes(self): return self.num_classes

class CleanEmotionDatasetFromArrays(Dataset):
    def __init__(self, feats, sizes, offsets, labels, class_names=None):
        self.feats = feats
        self.sizes = sizes
        self.offsets = offsets
        self.labels = labels
        # 如果没有提供class_names，使用默认的CASIA标签
        if class_names is None:
            self.class_names = ['angry', 'happy', 'neutral', 'sad']
        else:
            self.class_names = class_names
        self.num_classes = len(self.class_names)
    def __len__(self):
        return len(self.sizes)
    def __getitem__(self, index):
        offset = self.offsets[index]
        size = self.sizes[index]
        end = offset + size
        feats = torch.from_numpy(self.feats[offset:end, :].copy()).float()
        res = {"id": index, "feats": feats, "target": self.labels[index]}
        return res
    def collator(self, samples):
        if len(samples) == 0: return {}
        feats = [s["feats"] for s in samples]
        sizes = [s.shape[0] for s in feats]
        labels = torch.tensor([s["target"] for s in samples], dtype=torch.long)
        target_size = max(sizes)
        collated_feats = feats[0].new_zeros(len(feats), target_size, feats[0].size(-1))
        padding_mask = torch.BoolTensor(len(feats), target_size).fill_(False)
        for i, (feat, size) in enumerate(zip(feats, sizes)):
            collated_feats[i, :size] = feat
            padding_mask[i, size:] = True
        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {"feats": collated_feats, "padding_mask": padding_mask},
            "labels": labels
        }
        return res

def load_ssl_features(feature_path, label_dict, max_speech_seq_len=None):
    data, sizes, offsets, labels = load_emotion2vec_dataset(
        feature_path, labels='emo', min_length=1, max_length=max_speech_seq_len
    )
    if labels:
        numerical_labels = [label_dict[elem] for elem in labels]
    else:
        numerical_labels = []
    num = len(numerical_labels) if numerical_labels else len(sizes)
    iemocap_data = {
        "feats": data, "sizes": sizes, "offsets": offsets,
        "labels": numerical_labels, "num": num
    }
    return iemocap_data

# #############################################################################
# 重点修改：以下是新的、实现严格说话人隔离的数据加载逻辑
# #############################################################################

def _create_speaker_isolated_loaders(dataset, fold, session_samples, batch_size):
    """
    【核心函数】根据fold创建说话人隔离的训练、验证和测试集加载器。
    - 训练集: 3个sessions
    - 验证集: 1个session
    - 测试集: 1个session
    """
    feats = dataset['feats']
    sizes = dataset['sizes']
    offsets = dataset['offsets']
    labels = dataset['labels']
    
    num_sessions = len(session_samples)
    test_fold_index = fold
    # 验证集使用下一个session，实现环形交叉验证
    val_fold_index = (fold + 1) % num_sessions

    print("📊 [Clean Data] Speaker Isolation Strategy:")
    train_session_indices = [i + 1 for i in range(num_sessions) if i not in [test_fold_index, val_fold_index]]
    print(f"  - 🏋️  Training Sessions: {train_session_indices}")
    print(f"  - 🧐 Validation Session: {val_fold_index + 1}")
    print(f"  - 🧪 Test Session: {test_fold_index + 1}")

    session_starts = [0] + list(np.cumsum(session_samples))
    
    train_indices, val_indices, test_indices = [], [], []

    for i in range(num_sessions):
        start_idx = session_starts[i]
        end_idx = session_starts[i+1]
        session_indices = list(range(start_idx, end_idx))
        
        if i == test_fold_index:
            test_indices.extend(session_indices)
        elif i == val_fold_index:
            val_indices.extend(session_indices)
        else:
            train_indices.extend(session_indices)
    
    def create_subset(indices):
        """辅助函数：从索引列表创建数据集子集"""
        sub_labels = [labels[i] for i in indices]
        
        new_feats_list = []
        for i in indices:
            start = offsets[i]
            end = start + sizes[i]
            new_feats_list.append(feats[start:end, :])
            
        new_feats = np.concatenate(new_feats_list, axis=0)
        new_sizes = np.array([feat.shape[0] for feat in new_feats_list])
        new_offsets = np.concatenate([np.array([0]), np.cumsum(new_sizes)[:-1]], dtype=np.int64)

        return CleanEmotionDatasetFromArrays(new_feats, new_sizes, new_offsets, sub_labels)

    # 创建数据集
    train_dataset = create_subset(train_indices)
    val_dataset = create_subset(val_indices)
    test_dataset = create_subset(test_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collator, num_workers=0, pin_memory=False, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_dataset.collator, num_workers=0, pin_memory=False, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collator, num_workers=0, pin_memory=False, shuffle=False)
                            
    return train_loader, val_loader, test_loader

def create_clean_dataloaders(data_path, batch_size=None, fold=0):
    """
    【对外接口】创建干净数据的加载器，实现严格的5折说话人隔离。
    （原有的val_ratio参数已被移除，因为它不符合说话人隔离的原则）
    """
    if batch_size is None:
        batch_size = cfg.BATCH_SIZE
    
    print("📂 正在加载干净数据 (严格说话人隔离模式)...")
    
    dataset = load_ssl_features(data_path, cfg.LABEL_DICT)
    print(f"加载数据集，共 {dataset['num']} 个样本")
    
    n_samples = cfg.SESSION_SAMPLES
    
    train_loader, val_loader, test_loader = _create_speaker_isolated_loaders(
        dataset, fold, n_samples, batch_size
    )
    
    num_classes = len(cfg.LABEL_DICT)
    class_names = list(cfg.LABEL_DICT.keys())
    
    print(f"✅ 干净数据加载器创建完成 (Fold {fold+1}):")
    print(f"   - 训练批次数: {len(train_loader)} (来自 {len(train_loader.dataset)} 个样本)")
    print(f"   - 验证批次数: {len(val_loader)} (来自 {len(val_loader.dataset)} 个样本)")
    print(f"   - 测试批次数: {len(test_loader)} (来自 {len(test_loader.dataset)} 个样本)")
    print(f"   - 批次大小: {batch_size}")
    
    # 返回训练和验证加载器，以匹配 train.py 中的调用签名
    # test_loader 也可以返回，供最终测试使用
    return train_loader, val_loader, num_classes, class_names