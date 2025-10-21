import logging
import os
import contextlib

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

def get_session_ids(data_path, num_samples):
    """
    Reads the .emo file to extract session IDs for each utterance.
    Assumes the .emo file is in the same directory as the .npy file.
    """
    session_ids = []
    emo_file_path = os.path.splitext(data_path)[0] + ".emo"
    if not os.path.exists(emo_file_path):
        logger.error(f"Emotion file not found at: {emo_file_path}")
        # Return a dummy list of Nones if file not found
        return [None] * num_samples

    with open(emo_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # e.g., Ses01_impro01_F000_neu -> 1
            fname = line.split('\t')[0].strip()
            session_id = int(fname[4]) # 'Ses01...' -> 1
            session_ids.append(session_id)
    
    if len(session_ids) != num_samples:
        logger.warning(f"Number of session IDs ({len(session_ids)}) does not match number of samples ({num_samples}).")

    return session_ids

def get_fold_sessions(fold_id):
    """
    获取指定fold的训练、验证、测试会话
    
    Args:
        fold_id (int): fold编号 (1-5)
    
    Returns:
        tuple: (train_sessions, val_session, test_session)
    """
    if fold_id < 1 or fold_id > 5:
        raise ValueError(f"fold_id must be between 1 and 5, got {fold_id}")
    
    # 5折交叉验证的会话分配
    fold_configs = {
        1: ([1, 2, 3], 4, 5),  # Fold 1: 训练(1,2,3), 验证(4), 测试(5)
        2: ([2, 3, 4], 5, 1),  # Fold 2: 训练(2,3,4), 验证(5), 测试(1)
        3: ([3, 4, 5], 1, 2),  # Fold 3: 训练(3,4,5), 验证(1), 测试(2)
        4: ([4, 5, 1], 2, 3),  # Fold 4: 训练(4,5,1), 验证(2), 测试(3)
        5: ([5, 1, 2], 3, 4),  # Fold 5: 训练(5,1,2), 验证(3), 测试(4)
    }
    
    return fold_configs[fold_id]

def load_dataset(data_path, labels=None, min_length=3, max_length=None):
    sizes = []
    offsets = []
    emo_labels = []

    npy_data = np.load(data_path + ".npy")

    offset = 0
    skipped = 0

    if not os.path.exists(data_path + f".{labels}"):
        labels = None

    with open(data_path + ".lengths", "r") as len_f, open(
        data_path + f".{labels}", "r"
    ) if labels is not None else contextlib.ExitStack() as lbl_f:
        for line in len_f:
            length = int(line.rstrip())
            if labels is not None:
                lbl_line = next(lbl_f).rstrip()
                # 如果文件格式是每行只有一个情感标签，直接使用；否则分割后取第二个字段
                if '\t' in lbl_line:
                    lbl = lbl_line.split('\t')[1]  # 制表符分隔格式：文件名\t标签
                elif lbl_line.count(' ') == 0:
                    lbl = lbl_line  # 我们的格式：每行只有一个情感标签
                else:
                    lbl = lbl_line.split()[1]  # 原始格式：每行有多个字段
            else:
                lbl = None
            if length >= min_length and (
                max_length is None or length <= max_length
            ):
                sizes.append(length)
                offsets.append(offset)
                if lbl is not None:
                    emo_labels.append(lbl)
            offset += length

    sizes = np.asarray(sizes)
    offsets = np.asarray(offsets)

    logger.info(f"loaded {len(offsets)}, skipped {skipped} samples")

    return npy_data, sizes, offsets, emo_labels

class SpeechDataset(Dataset):
    def __init__(
        self,
        feats,
        sizes,
        offsets,
        labels=None,
        shuffle=True,
        sort_by_length=True,
    ):
        super().__init__()
        
        self.feats = feats
        self.sizes = sizes  # length of each sample
        self.offsets = offsets  # offset of each sample

        self.labels = labels

        self.shuffle = shuffle
        self.sort_by_length = sort_by_length

    def __getitem__(self, index):
        offset = self.offsets[index]
        end = self.sizes[index] + offset
        feats = torch.from_numpy(self.feats[offset:end, :].copy()).float()

        res = {"id": index, "feats": feats}
        if self.labels is not None:
            res["target"] = self.labels[index]

        return res

    def __len__(self):
        return len(self.sizes)

    def collator(self, samples):
        if len(samples) == 0:
            return {}

        feats = [s["feats"] for s in samples]
        sizes = [s.shape[0] for s in feats]
        labels = torch.tensor([s["target"] for s in samples]) if samples[0]["target"] is not None else None

        target_size = max(sizes)

        collated_feats = feats[0].new_zeros(
            len(feats), target_size, feats[0].size(-1)
        )

        padding_mask = torch.BoolTensor(torch.Size([len(feats), target_size])).fill_(False)
        for i, (feat, size) in enumerate(zip(feats, sizes)):
            collated_feats[i, :size] = feat
            padding_mask[i, size:] = True

        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {
                "feats": collated_feats,
                "padding_mask": padding_mask
            },
            "labels": labels
        }
        return res

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]

def load_ssl_features(feature_path, label_dict, max_speech_seq_len=None):
    # 构造正确的文件路径：feature_path/train
    data_path = os.path.join(feature_path, "train")
    
    data, sizes, offsets, labels = load_dataset(data_path, labels='emo', min_length=1, max_length=max_speech_seq_len)
    labels = [ label_dict[elem] for elem in labels ]
    
    num = len(labels)
    session_ids = get_session_ids(data_path, num)

    iemocap_data = {
        "feats": data,
        "sizes": sizes,
        "offsets": offsets,
        "labels": labels,
        "num": num,
        "session_ids": np.array(session_ids)
    } 

    return iemocap_data

def get_cv_dataloaders(data, batch_size, fold_id):
    """
    Creates DataLoaders for a specific fold of 5-fold cross-validation.
    每个fold划分为：3个训练会话、1个验证会话、1个测试会话
    
    Args:
        data (dict): The dataset dictionary from load_ssl_features.
        batch_size (int): The batch size.
        fold_id (int): The current fold for cross-validation (1 through 5).
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    feats = data['feats']
    sizes, offsets = data['sizes'], data['offsets']
    labels = np.array(data['labels'])
    session_ids = data['session_ids']

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
        
        # We need to rebuild the feature array for the subset to be contiguous
        sub_feats_list = []
        for idx in indices:
            start = data['offsets'][idx]
            end = start + data['sizes'][idx]
            sub_feats_list.append(feats[start:end, :])
        sub_feats_contiguous = np.concatenate(sub_feats_list, axis=0)
        
        sub_sizes = sizes[indices]
        sub_offsets = np.concatenate([np.array([0]), np.cumsum(sub_sizes)[:-1]], dtype=np.int64)

        print(f"   📁 {subset_name}: {len(indices)} samples")
        
        return SpeechDataset(
            feats=sub_feats_contiguous, 
            sizes=sub_sizes, 
            offsets=sub_offsets,
            labels=sub_labels,
        )

    train_dataset = create_subset(train_indices, "Training")
    val_dataset = create_subset(val_indices, "Validation")
    test_dataset = create_subset(test_indices, "Test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collator, 
                            num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_dataset.collator, 
                            num_workers=4, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collator, 
                            num_workers=4, pin_memory=True, shuffle=False)
    
    return train_loader, val_loader, test_loader