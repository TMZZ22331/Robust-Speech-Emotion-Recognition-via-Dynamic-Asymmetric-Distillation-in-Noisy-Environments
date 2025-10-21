#!/usr/bin/env python3
"""
IEMOCAP数据集跨域任务配置文件
从预训练的IEMOCAP权重开始，在0db噪声环境下进行跨域测试

🎯 任务设置：
   源域：IEMOCAP干净数据 (预训练权重)
   目标域：IEMOCAP 0db噪声数据 (跨域测试)
   
📊 数据配置：
   干净数据：processed_features_IEMOCAP
   噪声数据：processed_features_IEMOCAP_noisy/processed_features_noisy_0db
   
🔄 说话人隔离：
   5折交叉验证，严格按session隔离
   训练集：3个session
   验证集：1个session  
   测试集：1个session
"""

import torch
import os

# === 基础配置 ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42
USE_CUDA = True
GRADIENT_CLIPPING = True
MAX_GRAD_NORM = 1.0

# === IEMOCAP数据配置 ===
CLEAN_DATA_DIR = r"C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP"
NOISY_DATA_DIR = r"C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP_noisy\root1-babble-0db"
BATCH_SIZE = 64
NUM_WORKERS = 0
SESSION_SAMPLES = [1085, 1023, 1151, 1031, 1241]

# === IEMOCAP标签映射 ===
LABEL_DICT = {
    'ang': 0,
    'hap': 1,
    'neu': 2,
    'sad': 3
}

# 类别名称
CLASS_NAMES = ['ang', 'hap', 'neu', 'sad']
NUM_CLASSES = len(LABEL_DICT)

# === 预训练权重配置 ===
PRETRAINED_IEMOCAP_WEIGHT = r"C:\Users\admin\Desktop\111\good_-emo\IEMOCAP\pretrain-and-processed-IEMOCAP\train_for_clean_models\best_model_fold_2.ckpt"

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
DACP_QUALITY_SMOOTHING_BETA = 0.9  # 平滑系数 β

# 阶段三：阈值演化 (公式 12, 13)
DACP_SENSITIVITY_K = 10.0          # sigmoid 敏感系数 k
DACP_QUANTILE_START = 0.4      # 动态筛选标准起始值 q_start
DACP_QUANTILE_END = 0.80       # 动态筛选标准终止值 q_end

# 阶段四：最终阈值生成 (公式 15, 17)
DACP_CALIBRATION_STRENGTH_LAMBDA = 0.9 # 类别表现影响大小的校准强度 λ（0.3）
DACP_THRESHOLD_SMOOTHING_ALPHA = 0.9   # 最终阈值平滑系数 α

# === 消融实验和模块控制 ===
USE_DACP = True               # [消融开关] 是否启用DACP动态阈值
USE_ECDA = True               # [消融开关] 是否启用ECDA分布对齐
FIXED_CONFIDENCE_THRESHOLD = 0.9 # [消融开关] 当不使用DACP时，使用的固定置信度阈值

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
ECDA_COMPACTNESS_WEIGHT_GAMMA =0.1# 紧凑性损失权重 γ (降低以增加稳定性)
ECDA_REPULSION_WEIGHT_DELTA = 0.1 # 类间斥力损失权重 δ

# 损失权重dd
WEIGHT_CONSISTENCY = 1.0      # 一致性损失权重
TARGET_SCL_WEIGHT = 0.0       # SCL损失权重 (设置为0)
WEIGHT_ECDA = 0.3         # 新的ECDA损失权重 (λ_ECDA)

# === 训练配置 ===
EPOCHS = 500
LEARNING_RATE = 5e-4          # 跨域任务使用更低学习率（5e-4）
WEIGHT_DECAY = 1e-5
LEARNING_RATE_SCHEDULER = "cosine"
N_FOLDS = 2  # 5-fold cross-validation

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
MODEL_SAVE_DIR = "iemocap_cross_domain_models"
LOG_DIR = "iemocap_cross_domain_logs"
RESULTS_DIR = "iemocap_cross_domain_results"

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
    """验证IEMOCAP配置参数"""
    print("🔍 验证IEMOCAP跨域配置参数...")
    
    # 检查数据路径
    if not os.path.exists(os.path.dirname(CLEAN_DATA_DIR)):
        print(f"⚠️ 警告: IEMOCAP干净数据目录不存在: {CLEAN_DATA_DIR}")
    
    if not os.path.exists(os.path.dirname(NOISY_DATA_DIR)):
        print(f"⚠️ 警告: IEMOCAP噪声数据目录不存在: {NOISY_DATA_DIR}")
    
    # 检查预训练权重
    if PRETRAINED_IEMOCAP_WEIGHT and not os.path.exists(PRETRAINED_IEMOCAP_WEIGHT):
        print(f"⚠️ 警告: IEMOCAP预训练权重不存在: {PRETRAINED_IEMOCAP_WEIGHT}")
    
    # 检查参数合理性
    assert BATCH_SIZE > 0, "批次大小必须大于0"
    assert EPOCHS > 0, "训练轮次必须大于0"
    assert LEARNING_RATE > 0, "学习率必须大于0"
    assert NUM_CLASSES == 4, "IEMOCAP数据集应有4个类别"
    assert len(LABEL_DICT) == NUM_CLASSES, "标签字典长度与类别数不匹配"
    
    print("✅ IEMOCAP跨域配置参数验证完成")

def print_config():
    """打印IEMOCAP配置信息"""
    print("📋 IEMOCAP跨域任务配置:")
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
    print(f"   💾 预训练权重: {PRETRAINED_IEMOCAP_WEIGHT}")
    print(f"   📁 保存目录: {MODEL_SAVE_DIR}")

def setup_environment():
    """设置IEMOCAP训练环境"""
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
    
    print("🔧 IEMOCAP跨域训练环境设置完成")

if __name__ == "__main__":
    print("🚀 IEMOCAP跨域任务配置文件")
    print("=" * 60)
    
    setup_environment()
    validate_config()
    print_config()
    
    print("=" * 60)
    print("✅ IEMOCAP配置文件加载完成！") 