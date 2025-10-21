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

def get_session_ids(data_path, num_samples):
    """
    Reads the .emo file to extract session IDs for each utterance.
    """
    session_ids = []
    # The label file is specified by the 'labels' argument in load_emotion2vec_dataset,
    # which is hardcoded to 'emo'. So we look for a .emo file.
    emo_file_path = data_path + ".emo"
    if not os.path.exists(emo_file_path):
        logger.error(f"Emotion file not found at: {emo_file_path}")
        return [None] * num_samples

    with open(emo_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # e.g., Ses01_impro01_F000_neu -> 1
            fname = line.split('\t')[0].strip()
            session_id = int(fname[4])
            session_ids.append(session_id)
    
    if len(session_ids) != num_samples:
        logger.warning(f"Number of session IDs ({len(session_ids)}) does not match number of samples ({num_samples}).")

    return session_ids

def get_fold_sessions(fold_id):
    """
    获取指定fold的训练、验证、测试会话
    与预训练阶段保持一致的划分策略
    
    Args:
        fold_id (int): fold编号 (1-5)
    
    Returns:
        tuple: (train_sessions, val_session, test_session)
    """
    if fold_id < 1 or fold_id > 5:
        raise ValueError(f"fold_id must be between 1 and 5, got {fold_id}")
    
    # 5折交叉验证的会话分配（与预训练阶段完全一致）
    fold_configs = {
        1: ([1, 2, 3], 4, 5),  # Fold 1: 训练(1,2,3), 验证(4), 测试(5)
        2: ([2, 3, 4], 5, 1),  # Fold 2: 训练(2,3,4), 验证(5), 测试(1)
        3: ([3, 4, 5], 1, 2),  # Fold 3: 训练(3,4,5), 验证(1), 测试(2)
        4: ([4, 5, 1], 2, 3),  # Fold 4: 训练(4,5,1), 验证(2), 测试(3)
        5: ([5, 1, 2], 3, 4),  # Fold 5: 训练(5,1,2), 验证(3), 测试(4)
    }
    
    return fold_configs[fold_id]

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
    def __init__(self, feats, sizes, offsets, labels):
        self.feats = feats
        self.sizes = sizes
        self.offsets = offsets
        self.labels = labels
        self.class_names = list(cfg.LABEL_DICT.keys())
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
    # 修正：在这里统一拼接/train路径
    data_path = os.path.join(feature_path, "train")

    data, sizes, offsets, labels = load_emotion2vec_dataset(
        data_path, labels='emo', min_length=1, max_length=max_speech_seq_len
    )
    if labels:
        numerical_labels = [label_dict[elem] for elem in labels]
    else:
        numerical_labels = []
    
    # 修正：get_session_ids也需要基于拼接后的路径
    num = len(numerical_labels) if numerical_labels else len(sizes)
    session_ids = get_session_ids(data_path, num)

    iemocap_data = {
        "feats": data, "sizes": sizes, "offsets": offsets,
        "labels": numerical_labels, "num": num,
        "session_ids": np.array(session_ids)
    }
    return iemocap_data

# #############################################################################
# 重点修改：以下是新的、实现严格说话人隔离的数据加载逻辑
# #############################################################################

def get_cv_dataloaders(data_path, batch_size=None, fold_id=1):
    """
    Creates DataLoaders for a specific fold of 5-fold cross-validation.
    每个fold划分为：3个训练会话、1个验证会话、1个测试会话
    与预训练阶段保持一致的数据划分策略
    
    Args:
        data_path (str): 数据路径
        batch_size (int): 批次大小
        fold_id (int): fold编号 (1-5)
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names, num_classes)
    """
    if batch_size is None:
        batch_size = cfg.BATCH_SIZE
    
    print(f"📂 [Clean Data] Loading data for Fold {fold_id}...")
    
    # 注意：这里的data_path现在是根目录，load_ssl_features会处理/train
    dataset = load_ssl_features(data_path, cfg.LABEL_DICT)
    
    feats = dataset['feats']
    sizes = dataset['sizes']
    offsets = dataset['offsets']
    labels = np.array(dataset['labels'])
    session_ids = dataset['session_ids']

    # 获取当前fold的会话分配
    train_sessions, val_session, test_session = get_fold_sessions(fold_id)
    
    print(f"📊 Fold {fold_id} Session Assignment:")
    print(f"   🏋️  Training Sessions: {train_sessions}")
    print(f"   🧐 Validation Session: {val_session}")
    print(f"   🧪 Test Session: {test_session}")

    # 根据会话ID划分数据集
    train_indices = np.where(np.isin(session_ids, train_sessions))[0]
    val_indices = np.where(session_ids == val_session)[0]
    test_indices = np.where(session_ids == test_session)[0]

    def create_subset(indices, subset_name):
        """Helper to create a dataset subset from indices."""
        sub_labels = labels[indices].tolist()
        
        # This part is tricky. We need to create a contiguous feature array for the subset.
        sub_feats_list = [feats[offsets[i] : offsets[i] + sizes[i]] for i in indices]
        sub_feats = np.concatenate(sub_feats_list, axis=0)
        sub_sizes = sizes[indices]
        sub_offsets = np.concatenate([np.array([0]), np.cumsum(sub_sizes)[:-1]], dtype=np.int64)

        print(f"   📁 {subset_name}: {len(indices)} samples")
        
        return CleanEmotionDatasetFromArrays(sub_feats, sub_sizes, sub_offsets, sub_labels)

    train_dataset = create_subset(train_indices, "Training")
    val_dataset = create_subset(val_indices, "Validation")
    test_dataset = create_subset(test_indices, "Test")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collator, num_workers=0, pin_memory=False, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_dataset.collator, num_workers=0, pin_memory=False, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collator, num_workers=0, pin_memory=False, shuffle=False)
    
    num_classes = len(cfg.LABEL_DICT)
    class_names = list(cfg.LABEL_DICT.keys())

    print(f"✅ Clean data loaders created for Fold {fold_id}:")
    print(f"   🏋️  Training samples: {len(train_dataset)}")
    print(f"   🧐 Validation samples: {len(val_dataset)}")
    print(f"   🧪 Test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, class_names, num_classes