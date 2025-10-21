#!/usr/bin/env python3
"""
EmoDB数据集跨域任务配置文件
从预训练的EmoDB权重开始，在20db噪声环境下进行跨域测试

🎯 任务设置：
   源域：EmoDB干净数据 (预训练权重)
   目标域：EmoDB 20db噪声数据 (跨域测试)
   
📊 数据配置：
   干净数据：processed_features_EMODB_clean
   噪声数据：processed_features_EMODB_noisy/processed_features_noisy_20db
   
🔄 说话人隔离：
   10折交叉验证，严格按说话人隔离
   训练集：6个说话人
   验证集：2个说话人  
   测试集：2个说话人
"""

import torch
import os

# === 基础配置 ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42
USE_CUDA = True
GRADIENT_CLIPPING = True
MAX_GRAD_NORM = 1.0

# === EMODB数据配置 ===
CLEAN_DATA_DIR = r"C:\Users\admin\Desktop\DATA\processed_features_EMODB\processed_features_clean"
NOISY_DATA_DIR = r"C:\Users\admin\Desktop\DATA\processed_features_EMODB_noisy\root1-babble-10db"
BATCH_SIZE = 64
NUM_WORKERS = 0

# EMODB数据集特征路径（去掉.npy后缀）
CLEAN_FEAT_PATH = os.path.join(CLEAN_DATA_DIR, "train")
NOISY_FEAT_PATH = os.path.join(NOISY_DATA_DIR, "train")

# === EMODB标签映射 ===
LABEL_DICT = {
    'angry': 0,
    'happy': 1, 
    'neutral': 2,
    'sad': 3
}

# 类别名称
CLASS_NAMES = ['angry', 'happy', 'neutral', 'sad']
NUM_CLASSES = len(LABEL_DICT)

# === 预训练权重配置 ===
PRETRAINED_EMODB_WEIGHT = r"C:\Users\admin\Desktop\111\good_-emo\EMODB\pretrain-and-processed-EMODB\train_for_clean_models\emodb_loso_20250731_134905\best_model_fold_4.ckpt"

# === 模型配置 ===
INPUT_DIM = 768
HIDDEN_DIM = 256
DROPOUT_RATE = 0.1

# === SSRL跨域训练配置 ===
# 教师-学生网络
EMA_MOMENTUM = 0.995

# 训练策略
WARMUP_EPOCHS = 30            # 预热阶段
SCL_START_EPOCH = 5001          # SCL损失启动轮次 (设置为超大值以禁用)
ECDA_START_EPOCH = 30         # ECDA 启动轮次, 在教师网络稳定后再启动
DISABLE_MMD = True            # 禁用旧的MMD损失, 因为被ECDA替代

# === DACP (动态自适应置信度剪枝) 配置 ===
# 阶段二：类别表现追踪 (公式 10)
DACP_QUALITY_SMOOTHING_BETA = 0.8  # 平滑系数 β

# 阶段三：阈值演化 (公式 12, 13)
DACP_SENSITIVITY_K = 10.0          # sigmoid 敏感系数 k
DACP_QUANTILE_START = 0.4       # 动态筛选标准起始值 q_start
DACP_QUANTILE_END = 0.8         # 动态筛选标准终止值 q_end

# 阶段四：最终阈值生成 (公式 15, 17)
DACP_CALIBRATION_STRENGTH_LAMBDA = 0.3 # 类别表现影响大小的校准强度 λ
DACP_THRESHOLD_SMOOTHING_ALPHA = 0.9   # 最终阈值平滑系数 α

# === 消融实验和模块控制 ===
USE_DACP = True               # [消融开关] 是否启用DACP动态阈值
USE_ECDA = True               # [消融开关] 是否启用ECDA分布对齐
FIXED_CONFIDENCE_THRESHOLD = 0.75 # [消融开关] 当不使用DACP时，使用的固定置信度阈值

# === 新增消融实验配置参数 ===
USE_ENTROPY_IN_SCORE = True   # [消融开关] 是否在置信度分数中使用熵增强
USE_CLASS_AWARE_MMD = True    # [消融开关] 是否使用类别感知的MMD（否则使用全局MMD）

# === 锚点校准 (Anchor Calibration) 配置 ===
ANCHOR_CALIBRATION_ENABLED = True
ANCHOR_STD_K = 1.5 # 基准锚点标准差倍数 kσ (公式 5)

# === ECDA (能量感知分布对齐) 配置 ===
# 阶段二：类别级注意力 (公式 24)
ECDA_CLASS_ATTENTION_LAMBDA = 1.0 # 类别级注意力强度 λ_class

# 阶段三 & 四：紧凑性与斥力 (公式 26, 28)
ECDA_COMPACTNESS_WEIGHT_GAMMA = 0.1 # 紧凑性损失权重 γ (降低以增加稳定性)
ECDA_REPULSION_WEIGHT_DELTA = 0.1  # 类间斥力损失权重 δ

# 损失权重
WEIGHT_CONSISTENCY = 1.0      # 一致性损失权重
TARGET_SCL_WEIGHT = 0.0       # SCL损失权重 (设置为0)
WEIGHT_ECDA = 0.1          # 新的ECDA损失权重 (λ_ECDA)

# === 训练配置 ===
EPOCHS = 500
LEARNING_RATE = 5e-3          # 跨域任务使用更低学习率
WEIGHT_DECAY = 1e-5
LEARNING_RATE_SCHEDULER = "cosine"

# 正则化
USE_LABEL_SMOOTHING = True
LABEL_SMOOTHING_FACTOR = 0.05

# === 数据增强配置 ===
# 核心增强参数
WEAK_NOISE_STD = 0.01          # 教师网络弱增强噪声
STRONG_NOISE_STD = 0.05        # 学生网络强增强噪声
# DROPOUT_RATE = 0.1             # 已在模型配置中定义
TEMPORAL_MASK_RATIO = 0.1      # 时序遮盖比例

# 兼容性参数 (如果其他地方用到)
AUGMENT_DROPOUT_RATE = DROPOUT_RATE
FEATURE_DROPOUT_RATE = DROPOUT_RATE
NOISE_INJECTION_STD = STRONG_NOISE_STD

# === 验证和保存配置 ===
VALIDATION_INTERVAL = 5
PLOT_INTERVAL = 10
SAVE_BEST_MODEL = True
PLOT_CONFUSION_MATRIX = True

# 结果保存目录
MODEL_SAVE_DIR = "emodb_cross_domain_models"
LOG_DIR = "emodb_cross_domain_logs"
RESULTS_DIR = "emodb_cross_domain_results"

# === 渐进式训练策略 ===
PROGRESSIVE_TRAINING = True
INITIAL_CONSISTENCY_WEIGHT = 0.1
FINAL_CONSISTENCY_WEIGHT = 0.3
WEIGHT_RAMP_EPOCHS = 30

# 早停策略
EARLY_STOPPING = True
PATIENCE = 50                # 早停耐心值
MIN_DELTA = 0.001

# === 显示配置 ===
PRINT_MODEL_INFO = True
DEBUG_MODE = False
VERBOSE_LOGGING = True

# === 配置验证函数 ===
def validate_config():
    """验证EMODB配置参数"""
    print("🔍 验证EMODB跨域配置参数...")
    
    # 检查数据路径
    if not os.path.exists(CLEAN_DATA_DIR):
        print(f"⚠️ 警告: EMODB干净数据目录不存在: {CLEAN_DATA_DIR}")
    
    if not os.path.exists(NOISY_DATA_DIR):
        print(f"⚠️ 警告: EMODB噪声数据目录不存在: {NOISY_DATA_DIR}")
    
    # 检查预训练权重
    if not os.path.exists(PRETRAINED_EMODB_WEIGHT):
        print(f"⚠️ 警告: EMODB预训练权重不存在: {PRETRAINED_EMODB_WEIGHT}")
    
    # 检查特征文件
    clean_files = [f"{CLEAN_FEAT_PATH}.npy", f"{CLEAN_FEAT_PATH}.lengths", 
                   f"{CLEAN_FEAT_PATH}.lbl", f"{CLEAN_FEAT_PATH}.spk"]
    noisy_files = [f"{NOISY_FEAT_PATH}.npy", f"{NOISY_FEAT_PATH}.lengths", 
                   f"{NOISY_FEAT_PATH}.lbl", f"{NOISY_FEAT_PATH}.spk"]
    
    for file_path in clean_files:
        if not os.path.exists(file_path):
            print(f"⚠️ 警告: 干净数据特征文件不存在: {file_path}")
    
    for file_path in noisy_files:
        if not os.path.exists(file_path):
            print(f"⚠️ 警告: 噪声数据特征文件不存在: {file_path}")
    
    # 检查参数合理性
    assert BATCH_SIZE > 0, "批次大小必须大于0"
    assert EPOCHS > 0, "训练轮次必须大于0"
    assert LEARNING_RATE > 0, "学习率必须大于0"
    assert NUM_CLASSES == 4, "EMODB数据集应有4个类别"
    assert len(LABEL_DICT) == NUM_CLASSES, "标签字典长度与类别数不匹配"
    
    print("✅ EMODB跨域配置参数验证完成")

def print_config():
    """打印EMODB配置信息"""
    print("📋 EMODB跨域任务配置:")
    print(f"   🖥️  设备: {DEVICE}")
    print(f"   📊 批次大小: {BATCH_SIZE}")
    print(f"   🔢 训练轮次: {EPOCHS} (预热: {WARMUP_EPOCHS})")
    print(f"   📈 学习率: {LEARNING_RATE}")
    print(f"   🎯 模型维度: {INPUT_DIM} → {HIDDEN_DIM} → {NUM_CLASSES}")
    print(f"   🏷️  标签映射: {LABEL_DICT}")
    print(f"   🔄 EMA动量: {EMA_MOMENTUM}")
    print(f"   ⚖️  损失权重: SCL={TARGET_SCL_WEIGHT}, ECDA={WEIGHT_ECDA}")
    print(f"   🗂️  数据路径:")
    print(f"      - 干净数据: {CLEAN_DATA_DIR}")
    print(f"      - 噪声数据: {NOISY_DATA_DIR}")
    print(f"   💾 预训练权重: {PRETRAINED_EMODB_WEIGHT}")
    print(f"   📁 保存目录: {MODEL_SAVE_DIR}")

def setup_environment():
    """设置EMODB训练环境"""
    # 创建必要目录
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 设置随机种子
    import random
    import numpy as np
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    print("🔧 EMODB跨域训练环境设置完成")

if __name__ == "__main__":
    print("🚀 EMODB跨域任务配置文件")
    print("=" * 60)
    
    setup_environment()
    validate_config()
    print_config()
    
    print("=" * 60)
    print("✅ EMODB配置文件加载完成！") 