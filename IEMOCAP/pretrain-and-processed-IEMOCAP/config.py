# 训练配置文件
# Training Configuration for IEMOCAP Emotion Recognition

class TrainingConfig:
    """训练配置类"""
    
    # 数据路径配置
    FEAT_PATH = "C:/Users/admin/Desktop/DATA/processed_features_IEMOCAP"
    
    # 模型配置
    INPUT_DIM = 768  # emotion2vec特征维度
    OUTPUT_DIM = 4   # 四分类：ang, hap, neu, sad
    
    # 训练配置
    MAX_EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-5
    
    # 早停配置
    EARLY_STOPPING_PATIENCE = 20
    EARLY_STOPPING_MIN_DELTA = 0.001
    EARLY_STOPPING_METRIC = "val_weighted_acc"  # "val_loss", "val_acc", "val_weighted_acc", "val_f1"
    EARLY_STOPPING_MODE = "max"  # "min" for loss, "max" for accuracy/f1
    
    # 学习率调度器配置
    LR_SCHEDULER_TYPE = "ReduceLROnPlateau"  # "ReduceLROnPlateau", "CosineAnnealingWarmRestarts", "StepLR"
    LR_SCHEDULER_FACTOR = 0.7
    LR_SCHEDULER_PATIENCE = 8
    LR_SCHEDULER_MIN_LR = 1e-6
    
    # 余弦退火配置
    COSINE_T_0 = 10  # 第一次重启的周期
    COSINE_T_MULT = 2  # 周期倍数
    COSINE_ETA_MIN = 1e-6  # 最小学习率
    
    # 验证集配置
    VALIDATION_RATIO = 0.1  # 从训练集中分出10%作为验证集
    
    # 保存配置
    SAVE_DIR = "train_for_clean_models"
    
    # 数据集名称（用于文件命名）
    DATASET_NAME = "iemocap"
    
    # IEMOCAP数据集配置 (使用数据中的实际标签)
    LABEL_DICT = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
    SESSION_SAMPLES = [1085, 1023, 1151, 1031, 1241] # CASIA各说话人样本数 (IEMOCAP:SESSION_SAMPLES = [1085, 1023, 1151, 1031, 1241]，CASIA:SESSION_SAMPLES = [1499, 1500, 1499, 1498] )
    
    # GPU配置
    USE_CUDA = True
    CUDA_BENCHMARK = True  # 优化CUDA性能
    
    # 随机种子
    RANDOM_SEED = 42
    
    # 显示配置
    PRINT_EVERY_N_EPOCHS = 5  # 每N个epoch显示一次进度
    GPU_MONITOR_EVERY_N_EPOCHS = 10  # 每N个epoch监控一次GPU
    
    @classmethod
    def get_config_dict(cls):
        """获取配置字典"""
        config = {}
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)) and attr != 'get_config_dict':
                config[attr] = getattr(cls, attr)
        return config
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 60)
        print("Training Configuration")
        print("=" * 60)
        config = cls.get_config_dict()
        for key, value in config.items():
            print(f"{key:30s}: {value}")
        print("=" * 60)


# 可选的高级配置
class AdvancedConfig(TrainingConfig):
    """高级配置 - 可以继承TrainingConfig并修改特定参数"""
    
    # 更保守的训练配置（更长的patience，更小的学习率）
    EARLY_STOPPING_PATIENCE = 30
    LEARNING_RATE = 1e-4
    LR_SCHEDULER_PATIENCE = 12
    
    # 使用余弦退火策略
    LR_SCHEDULER_TYPE = "CosineAnnealingWarmRestarts"
    COSINE_T_0 = 15
    COSINE_T_MULT = 2
    COSINE_ETA_MIN = 5e-7
    
    # 更大的batch size（如果GPU内存充足）
    BATCH_SIZE = 128
    
    # 更严格的早停条件（基于加权准确率）
    EARLY_STOPPING_MIN_DELTA = 0.001
    EARLY_STOPPING_METRIC = "val_weighted_acc"
    EARLY_STOPPING_MODE = "max"


# 余弦退火配置
class CosineConfig(TrainingConfig):
    """余弦退火配置 - 使用余弦退火策略和accuracy早停"""
    
    # 余弦退火策略
    LR_SCHEDULER_TYPE = "CosineAnnealingWarmRestarts"
    LEARNING_RATE = 3e-4  # 稍高的初始学习率
    COSINE_T_0 = 12  # 第一次重启周期
    COSINE_T_MULT = 2
    COSINE_ETA_MIN = 1e-7
    
    # 基于加权准确率的早停
    EARLY_STOPPING_METRIC = "val_weighted_acc"
    EARLY_STOPPING_MODE = "max"
    EARLY_STOPPING_PATIENCE = 25  # 更长的patience
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    MAX_EPOCHS = 120  # 更多epochs配合余弦退火
    

# EmoDB数据集配置
class EmoDBConfig(TrainingConfig):
    """EmoDB数据集配置 - 针对EmoDB数据集的样本划分"""
    
    # EmoDB数据集的样本划分 (总共291个样本)
    SESSION_SAMPLES = [58, 58, 58, 58, 59]  # 5折交叉验证，每fold约58个样本
    
    # EmoDB专用保存目录 - 使用时间戳
    SAVE_DIR = "train_for_clean_models"  # 基础目录，实际使用时会添加时间戳
    
    # EmoDB数据集标识符，用于文件命名
    DATASET_NAME = "emodb"


# 开发/调试配置
class DebugConfig(EmoDBConfig):
    """调试配置 - 快速验证代码"""
    
    MAX_EPOCHS = 10  # 只训练10轮进行快速测试
    EARLY_STOPPING_PATIENCE = 3
    PRINT_EVERY_N_EPOCHS = 1  # 每个epoch都显示
    GPU_MONITOR_EVERY_N_EPOCHS = 2 