# 🎭 跨域情感识别：半监督学习与说话人隔离

## 📋 项目概述

本项目实现了一个基于**半监督学习**和**严格说话人隔离**的跨域情感识别系统，支持三个多语言数据集：IEMOCAP（英语）、CASIA（中文）和EMODB（德语）。项目采用**教师-学生网络**架构，实现从干净语音到噪声语音的域适应。

##
**权重emotion2vec_base.pt，是所有预处理的基础权重**

**权重emotion2vec_base.pt，是emotion2vec公布的base版本的权重，这是其Git hub官方链接[emotion2vec](https://github.com/ddlBoJack/emotion2vec?tab=readme-ov-file)**

其官方公布的权重下载链接是[Baidu Netdisk](https://pan.baidu.com/s/15zqmNTYa0mkEwlIom7DO3g?pwd=b9fq)(password: b9fq)
##

-针对于fairseq库问题，首先pip需要先降级到24.0,然后在项目的工作空间根目录里执行命令：git clone https://github.com/pytorch/fairseq
 cd fairseq
 pip install --editable ./

### 🎯 核心特性
- **🏫 教师-学生网络**：教师网络弱增强+EMA更新，学生网络强增强+梯度更新
- **🌍 多语言支持**：支持英语、中文、德语三种语言的情感识别
- **📊 两阶段训练**：预训练阶段获得基础权重，跨域训练阶段进行域适应

## 🏗️ 项目结构

```
/
├── README.md                           # 总体项目说明
├── IEMOCAP/                            # 英语数据集
│   ├── README.md                       # IEMOCAP详细说明
│   ├── pretrain-and-processed-IEMOCAP/ # 预训练版本
│   │   ├── scripts/                    # 预处理脚本
│   │   ├── config.py                   # 预训练配置
│   │   ├── train_for_clean.py          # 监督预训练
│   │   └── emotion2vec_base.pt         # 预训练权重
│   └── DAD-train-IEMOCAP/              # 跨域训练版本
│       ├── config.py                   # 跨域训练配置
│       ├── dataload_clean.py           # 干净数据加载器
│       ├── dataload_noisy.py           # 噪声数据加载器
│       ├── model.py                    # SSRL模型
│       ├── train.py                    # 主训练脚本
│       └── utils.py                    # 工具函数
├── CASIA/                              # 中文数据集
│   ├── README.md                       # CASIA详细说明
│   ├── pretrain-and-processed-CASIA/   # 预训练版本
│   └── DAD-train-CASIA/                # 跨域训练版本
│       └── (类似IEMOCAP结构)
└── EMODB/                              # 德语数据集
    ├── README.md                       # EMODB详细说明
    ├── pretrain-and-processed-EMODB/   # 预训练版本
    └── DAD-train-EMODB/                # 跨域训练版本
        └── (类似IEMOCAP结构)
```

## 📊 数据集对比

| 数据集 | 语言 | 样本数 | 说话人数 | 交叉验证 | 隔离策略 | 特色 |
|--------|------|--------|----------|----------|----------|------|
| **IEMOCAP** | 英语 | 5,531 | 5 Sessions | 5折 | 按Session隔离 | 多模态、会话式 |
| **CASIA** | 中文 | 5,996 | 4人 | 4折 | 按说话人隔离 | 中文声调、文化特色 |
| **EMODB** | 德语 | ~500 | 10人 | 5折 | 按说话人分组隔离 | 小数据集、高质量 |

### 🎯 情感类别
所有数据集统一使用4类情感：**愤怒 (angry)**、**快乐 (happy)**、**中性 (neutral)**、**悲伤 (sad)**

## 🔧 技术架构

### 1. 特征提取
- **基础特征**：使用预训练的emotion2vec模型提取768维特征
- **音频预处理**：统一采样率、噪声添加、特征标准化

### 2. 说话人隔离策略
```python
# IEMOCAP: 5折交叉验证
test_session = fold                    # 测试集：1个Session
val_session = (fold + 1) % 5          # 验证集：1个Session  
train_sessions = [其余3个Sessions]     # 训练集：3个Sessions

# CASIA: 4折交叉验证
test_speaker = speakers[fold]          # 测试集：1个说话人
val_speaker = speakers[(fold+1) % 4]   # 验证集：1个说话人
train_speakers = [其余2个说话人]       # 训练集：2个说话人

# EMODB: 5折交叉验证
test_speakers = groups[fold]           # 测试集：2个说话人
val_speakers = groups[(fold+1) % 5]    # 验证集：2个说话人
train_speakers = [其余6个说话人]       # 训练集：6个说话人
```

### 3. 半监督学习架构
```python
# 教师-学生网络
class SSRLModel:
    def __init__(self):
        self.student_encoder = Encoder()      # 学生编码器
        self.student_classifier = Classifier() # 学生分类器
        self.teacher_encoder = Encoder()      # 教师编码器（EMA）
        self.teacher_classifier = Classifier() # 教师分类器（EMA）

# 损失函数组合
total_loss = (
    supervised_loss +           # 监督损失（干净数据）
    consistency_loss +          # 一致性损失（噪声数据）
    scl_loss +                 # 监督对比损失
    mmd_loss                   # 最大均值差异损失
)
```

### 4. 数据增强策略
- **教师网络（弱增强）**：仅噪声注入
- **学生网络（强增强）**：噪声注入 + 特征dropout + 时序遮盖

## 🚀 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install torch torchaudio numpy scipy scikit-learn

# 克隆项目
git clone <repository-url>
cd 三个数据集的代码美化版本
```

### 2. 选择数据集
根据需要选择一个数据集进行实验：

#### IEMOCAP（英语）
```bash
cd IEMOCAP/
# 查看详细说明
cat README.md

# 预处理 → 预训练 → 跨域训练
cd pretrain-and-processed-IEMOCAP/scripts/
./complete_preprocessing.ps1
cd ..
python train_for_clean.py
cd ../DAD-train-IEMOCAP/
python train.py --fold 0
```

#### CASIA（中文）
```bash
cd CASIA/
# 查看详细说明
cat README.md

# 预处理 → 预训练 → 跨域训练
cd pretrain-and-processed-CASIA/scripts/
./casia_preprocessing.ps1
cd ..
python train_casia.py
cd ../DAD-train-CASIA/
python train_CASIA.py --fold 0
```

#### EMODB（德语）
```bash
cd EMODB/
# 查看详细说明
cat README.md

# 预处理 → 预训练 → 跨域训练
cd pretrain-and-processed-EMODB/scripts/
./emodb_preprocessing.ps1
cd ..
python run_training_emodb.py
cd ../DAD-train-EMODB/
python train_emodb.py --fold 0
```

## 📈 实验流程

### 阶段1：数据预处理
1. **音频预处理**：格式转换、采样率统一
2. **特征提取**：emotion2vec特征提取
3. **噪声添加**：生成不同SNR的噪声数据
4. **清单生成**：创建训练清单文件

### 阶段2：监督预训练
1. **模型初始化**：基于emotion2vec预训练权重
2. **监督学习**：使用干净数据进行监督训练
3. **交叉验证**：严格说话人隔离的K折验证
4. **权重保存**：保存每折的最佳权重

### 阶段3：跨域训练
1. **权重加载**：加载预训练权重
2. **双域训练**：同时使用干净和噪声数据
3. **半监督学习**：噪声数据无标签训练
4. **域适应**：教师-学生网络域适应

## 🔍 核心技术

### 1. 严格说话人隔离
- **目标**：避免训练集和测试集之间的说话人重叠
- **实现**：预定义说话人分组，确保完全分离
- **验证**：代码中包含断言检查，确保无重叠

### 2. 半监督学习
- **有标签数据**：干净语音数据，用于监督学习
- **无标签数据**：噪声语音数据，用于半监督学习
- **一致性正则化**：教师-学生网络一致性损失

### 3. 教师-学生网络
- **教师网络**：通过EMA更新，提供稳定的伪标签
- **学生网络**：通过梯度更新，学习噪声域特征
- **置信度筛选**：仅使用高置信度样本进行训练

### 4. 多阶段训练
- **预热阶段**：仅使用监督损失稳定训练
- **微调阶段**：加入半监督损失进行域适应

## 📊 评估指标

### 主要指标
- **准确率 (Accuracy)**：总体分类准确率
- **加权准确率 (Weighted Accuracy)**：平衡准确率
- **F1分数 (F1-Score)**：宏平均、微平均、加权平均
- **每类精确率/召回率**：各情感类别的详细指标

### 可视化分析
- **混淆矩阵**：分类结果详细分析
- **训练曲线**：损失和准确率变化曲线
- **类别分布**：各情感类别的分布情况

## 💡 使用建议

### 1. 参数调优
- **学习率**：建议从5e-5开始，根据数据集大小调整
- **批次大小**：IEMOCAP/CASIA使用64，EMODB使用32
- **预热期**：IEMOCAP/CASIA使用30轮，EMODB使用10轮

### 2. 数据准备
- **数据质量**：确保音频质量良好，避免严重失真
- **标签准确性**：检查情感标签的准确性和一致性
- **噪声设置**：根据实际应用场景设置合适的SNR

### 3. 实验设计
- **完整流程**：按照预处理→预训练→跨域训练的顺序执行
- **消融实验**：可以分别验证各个组件的贡献
- **多折验证**：运行完整的交叉验证获得可靠结果

## 📝 相关论文

### 基础技术
- **Emotion2vec**: Ma et al., "Emotion2vec: Self-supervised pre-training for speech emotion representation"
- **Mean Teacher**: Tarvainen & Valpola, "Mean teachers are better role models"
- **FixMatch**: Sohn et al., "FixMatch: Simplifying semi-supervised learning with consistency and confidence"

### 数据集
- **IEMOCAP**: Busso et al., "IEMOCAP: interactive emotional dyadic motion capture database"
- **CASIA**: "CASIA Chinese Affective Speech Database"
- **EMODB**: Burkhardt et al., "A database of German emotional speech"

### 应用领域
- **跨域适应**: 域适应和迁移学习相关研究
- **情感识别**: 语音情感识别和多模态情感分析
- **半监督学习**: 半监督学习在语音识别中的应用

## 🤝 贡献指南

欢迎提交问题和改进建议！请遵循以下步骤：

1. **Fork项目**并创建特性分支
2. **提交更改**并添加详细说明
3. **确保代码质量**：运行测试并检查代码风格
4. **提交Pull Request**并描述改进内容

## 📄 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至 [your-email@example.com]

---

**Happy Coding! 🎉** 