# EMODB 德语情感识别跨域实验

## 📊 数据集概述

**EMODB (Berlin Database of Emotional Speech)**是一个德语情感语音数据库，包含10个不同的说话人。

- **数据规模**：总计约500个样本
- **说话人分布**：10个说话人 (ID: 03, 08, 09, 10, 11, 12, 13, 14, 15, 16)
- **情感类别**：4类情感 (angry, happy, neutral, sad)
- **特征维度**：768维 (emotion2vec特征)
- **语言**：德语

## 🏗️ 代码结构

```
EMODB/
├── pretrain-and-processed-EMODB/      # 预训练版本
│   ├── scripts/                       # 预处理脚本
│   ├── config.py                      # 预训练配置
│   ├── run_training_emodb.py          # 监督预训练
│   └── emotion2vec_base.pt            # 预训练权重
└── DAD-train-EMODB/                   # 跨域训练版本
    ├── config_emodb.py                # 跨域训练配置
    ├── dataload_emodb_clean.py        # 干净数据加载器
    ├── dataload_emodb_noisy.py        # 噪声数据加载器
    ├── model.py                       # SSRL模型
    ├── train_emodb.py                 # 主训练脚本
    └── utils.py                       # 工具函数
```

## 🔧 完整实验流程

### 1. 数据预处理阶段

**位置**: `pretrain-and-processed-EMODB/scripts/`

#### 1.1 音频预处理
```bash
# 运行预处理脚本
./emodb_preprocessing.ps1
 # 示例1: 只处理干净数据
.\emodb_preprocessing.ps1 -EMODB_ROOT "C:\Users\admin\Desktop\DATA\EmoDB\EmoDB Dataset_wav_datasets" -OutputBasePath "C:\Users\admin\Desktop\DATA\meihua\EMODB"

 # 示例2: 处理多种信噪比的噪声数据 (0dB, 5dB, 10dB)
.\emodb_preprocessing.ps1 -EMODB_ROOT "C:\Users\admin\Desktop\DATA\EmoDB\EmoDB Dataset_wav_datasets" -OutputBasePath "C:\Users\admin\Desktop\DATA\meihua\EMODB" -AddNoise -SnrLevels @(20)



```



**预处理产物**：
- `train.npy` - 特征文件
- `train.lengths` - 序列长度
- `train.lbl` - 情感标签
- `train.spk` - 说话人ID（用于隔离）

### 2. 监督预训练阶段

**位置**: `pretrain-and-processed-EMODB/`

#### 2.1 配置参数
```python
# config.py 关键参数
INPUT_DIM = 768
OUTPUT_DIM = 4
MAX_EPOCHS = 100
BATCH_SIZE = 32  # 较小的batch size适应小数据集
LEARNING_RATE = 2e-4
```

#### 2.2 执行预训练
```bash
cd pretrain-and-processed-EMODB/
python run_training_emodb.py
```

**预训练特点**：
- 5折交叉验证（按说话人分组隔离）
- 纯监督学习
- 早停机制
- 学习率调度
- 针对小数据集的特殊优化

**预训练产物**：
- `train_for_clean_models/best_model_fold_X.ckpt` - 每折最佳权重

### 3. 跨域主训练阶段

**位置**: `DAD-train-EMODB/`

#### 3.1 配置参数
```python
# config_emodb.py 关键参数
CLEAN_DATA_DIR = "path/to/clean/data"
NOISY_DATA_DIR = "path/to/noisy/data"
PRETRAINED_EMODB_WEIGHT = "path/to/pretrained/model.ckpt"

# SSRL框架参数
WARMUP_EPOCHS = 10  # 较短的预热期
EMA_MOMENTUM = 0.995
CONFIDENCE_PERCENTILE = 70
WEIGHT_CONSISTENCY = 0.5
BATCH_SIZE = 32  # 适应小数据集
```

#### 3.2 执行跨域训练
```bash
cd DAD-train-EMODB/
python train_emodb.py --fold 0  # 指定折数 (0-4)
```

**跨域训练特点**：
- **严格说话人隔离**：训练、验证、测试集完全按说话人分组隔离
- **半监督学习**：
  - 干净数据：有标签监督学习
  - 噪声数据：无标签半监督学习
- **教师-学生网络**：
  - 教师网络：弱增强 + EMA更新
  - 学生网络：强增强 + 梯度更新
- **多阶段训练**：
  - 预热阶段：仅监督学习
  - 微调阶段：监督 + 半监督

## 🎯 说话人隔离策略

### 5折交叉验证设计
```python
# 预定义说话人分组
EMODB_SPEAKER_GROUPS = [
    ['03', '08'],  # Fold 1: 2个说话人
    ['09', '10'],  # Fold 2: 2个说话人
    ['11', '12'],  # Fold 3: 2个说话人
    ['13', '14'],  # Fold 4: 2个说话人
    ['15', '16']   # Fold 5: 2个说话人
]

# 以fold=0为例
test_speakers = ['03', '08']      # 测试集：2个说话人
val_speakers = ['09', '10']       # 验证集：2个说话人
train_speakers = ['11','12','13','14','15','16']  # 训练集：6个说话人
```

### 隔离验证
- ✅ 训练集说话人与测试集说话人完全分离
- ✅ 验证集说话人与测试集说话人完全分离
- ✅ 干净数据与噪声数据使用相同隔离策略
- ✅ 每折测试集包含2个说话人，确保充分的测试样本

## 🔍 半监督学习实现

### 数据标签处理
```python
# 干净数据：有标签
clean_dataset = CleanEmotionDatasetFromArrays(
    feats, sizes, offsets, labels, has_labels=True
)

# 噪声数据：无标签
noisy_dataset = NoisyEmotionDatasetFromArrays(
    feats, sizes, offsets, labels=None, has_labels=False
)
```

### 损失函数组合
```python
# 监督损失（干净数据）
supervised_loss = CrossEntropyLoss(clean_outputs, clean_labels)

# 一致性损失（噪声数据）
consistency_loss = KLDivLoss(student_outputs, teacher_outputs)


```

## 📈 实验结果

### 预训练结果
- 5折交叉验证平均准确率
- 各情感类别F1分数
- 混淆矩阵分析

### 跨域结果
- 干净→噪声域适应性能
- 不同SNR条件下的鲁棒性
- 消融实验结果

## 💡 使用建议

1. **首次使用**：先运行预处理脚本，确保数据格式正确
2. **预训练**：使用监督学习获得基础权重
3. **跨域训练**：基于预训练权重进行半监督域适应
4. **小数据集优化**：
   - 使用较小的batch size (32)
   - 较短的预热期 (10 epochs)
   - 更频繁的验证和早停
   - 适当的数据增强避免过拟合

## 🌍 德语特色

EMODB数据集作为德语情感语音数据库，具有以下特点：
- **语音特征**：德语特有的音素和韵律特征
- **表达方式**：德语情感表达的文化特色
- **跨语言研究**：可用于研究跨语言情感识别
- **小数据集挑战**：适合研究小样本学习方法

## 📊 数据集特点

### 优势
- **高质量录制**：专业录音棚环境
- **情感丰富**：表达清晰的情感类别
- **说话人多样性**：10个不同的说话人
- **标准化**：广泛使用的基准数据集

### 挑战
- **数据量小**：相比其他数据集样本较少
- **类别不平衡**：某些情感类别样本偏少
- **语言特异性**：德语特有的语言特征

## 📝 相关论文

- EMODB数据集：Burkhardt et al., "A database of German emotional speech"
- Emotion2vec特征：Ma et al., "Emotion2vec: Self-supervised pre-training for speech emotion representation"
- 半监督学习：Tarvainen & Valpola, "Mean teachers are better role models"
- 小样本学习：相关小数据集情感识别研究 