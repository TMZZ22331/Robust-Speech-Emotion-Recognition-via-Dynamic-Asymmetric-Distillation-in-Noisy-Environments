import os
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from collections import Counter
import contextlib
import logging
import json
from datetime import datetime

# ==========================================================================================
# 🔧 1. 配置区域：请在此处修改权重和数据路径
# ==========================================================================================
# 预训练权重路径 (通常是 emotion2vec_base.pt 或您在干净数据上训练的模型的 .ckpt)
PRETRAINED_WEIGHTS_PATH = r"C:\Users\admin\Desktop\论文参考\最终的代码副本\最终代码-截至7.30号副本\第一篇最终的代码\good_-emo\IEMOCAP\pretrain-and-processed-IEMOCAP\train_for_clean_models\best_model_fold_2.ckpt"

# 主干训练后的权重路径 (在噪声数据上微调后的模型)
# 脚本会自动从这个路径中提取dB信息来设置图表标题
FINETUNED_WEIGHTS_PATH = r"C:\Users\admin\Desktop\论文参考\最终的代码副本\最终代码-截至7.30号副本\第一篇最终的代码\iemocap_mutil-noisy_cross_domain_results\root1\babble\10db\fold_2\models\iemocap_cross_domain_best.pth"

# 包含噪声特征的目录路径 (包含 train.npy, train.lengths, train.emo 文件)
# 请确保这个路径指向您想要可视化的特定噪声等级的数据
NOISY_DATA_ROOT_PATH = r"C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP_noisy\root1-babble-10db"
# ==========================================================================================


# 日志和设备配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DEVICE = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
LABEL_DICT = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
CLASS_NAMES = list(LABEL_DICT.keys())
# 参考您的图片，使用相同的颜色映射：红、绿、蓝、橙
EMOTION_COLORS = {
    'ang': '#d62728',    # 红色 - Anger
    'hap': '#2ca02c',    # 绿色 - Happiness  
    'neu': '#1f77b4',    # 蓝色 - Neutral
    'sad': '#ff7f0e'     # 橙色 - Sadness
}
EMOTION_LABELS = {
    'ang': 'Anger',
    'hap': 'Happiness', 
    'neu': 'Neutral',
    'sad': 'Sadness'
}


# ==========================================================================================
# 模型定义 (从 good_-emo/IEMOCAP/DAD-train-IEMOCAP/model.py 复制)
# 使此脚本可以独立运行，无需依赖项目其他文件
# ==========================================================================================
class Emotion2VecEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, pretrained_path=None):
        super().__init__()
        self.pre_net = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.activate = nn.ReLU()
    def forward(self, x, padding_mask=None):
        x = self.activate(self.pre_net(x))
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).float())
            valid_lengths = (1 - padding_mask.float()).sum(dim=1, keepdim=True)
            x = x.sum(dim=1) / torch.clamp(valid_lengths, min=1.0)
        else:
            x = x.mean(dim=1)
        return x

class EmotionClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_layer = nn.Linear(in_features=input_dim, out_features=num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        logits = self.fc_layer(x)
        return logits

class SSRLModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=4):
        super().__init__()
        self.student_encoder = Emotion2VecEncoder(input_dim, hidden_dim)
        self.student_classifier = EmotionClassifier(hidden_dim, num_classes)
        # The t-SNE script does not need the teacher network, EMA updates, or loss calculations.
        # We only need the student network structure for loading weights and extracting embeddings.
    
    def get_embeddings(self, x: torch.Tensor, padding_mask=None):
        return self.student_encoder(x, padding_mask)

    def forward(self, x: torch.Tensor, padding_mask=None):
        embedding = self.get_embeddings(x, padding_mask)
        return self.student_classifier(embedding)

# ==========================================================================================
# 数据加载器 (专为 t-SNE 设计，始终加载标签)
# ==========================================================================================
def load_emotion2vec_for_tsne(data_path, labels='emo', min_length=1, max_length=None):
    """
    修改版加载器：始终尝试加载标签，用于可视化。
    """
    sizes, offsets, emo_labels = [], [], []
    npy_data = np.load(data_path + ".npy")
    offset, skipped = 0, 0
    label_file_path = data_path + f".{labels}"
    
    if not os.path.exists(label_file_path):
        raise FileNotFoundError(f"标签文件未找到: {label_file_path}。t-SNE需要标签进行着色。")

    with open(data_path + ".lengths", "r") as len_f, open(label_file_path, "r") as lbl_f:
        for line_idx, (len_line, lbl_line) in enumerate(zip(len_f, lbl_f)):
            length = int(len_line.rstrip())
            # 假设标签在每行的第二列，例如 "filename neu" -> "neu"
            lbl_parts = lbl_line.rstrip().split()
            if len(lbl_parts) < 2:
                logger.warning(f"标签文件第 {line_idx+1} 行格式不正确，已跳过：'{lbl_line.rstrip()}'")
                offset += length
                skipped += 1
                continue
            lbl = lbl_parts[1]
            
            if length >= min_length and (max_length is None or length <= max_length):
                sizes.append(length)
                offsets.append(offset)
                emo_labels.append(lbl)
            else:
                skipped += 1
            offset += length

    sizes, offsets = np.asarray(sizes), np.asarray(offsets)
    logger.info(f"成功加载 {len(offsets)} 个样本用于 t-SNE，跳过了 {skipped} 个样本。")
    return npy_data, sizes, offsets, emo_labels

class IEMOCAP_tSNE_Dataset(Dataset):
    def __init__(self, data_root_path):
        # 假设数据文件名为 'train.npy', 'train.lengths', 'train.emo'
        data_file_path = os.path.join(data_root_path, "train")
        logger.info(f"从 {data_file_path} 加载数据...")
        
        self.feats, self.sizes, self.offsets, self.labels = load_emotion2vec_for_tsne(data_file_path)
        
        if not self.labels:
            raise ValueError("未能从数据文件中加载任何标签。")
            
        self.numerical_labels = [LABEL_DICT[label] for label in self.labels]
        self.class_counts = Counter(self.labels)
        
        logger.info(f"数据集加载完成。共 {len(self)} 个样本。")
        for label, count in self.class_counts.items():
            logger.info(f"  - 类别 '{label}': {count} 个样本")

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, index):
        offset = self.offsets[index]
        size = self.sizes[index]
        end = offset + size
        feats = torch.from_numpy(self.feats[offset:end, :].copy()).float()
        return {"feats": feats, "target": self.numerical_labels[index]}

    def collator(self, samples):
        if len(samples) == 0: return {}
        feats = [s["feats"] for s in samples]
        # 修复: 此前这里错误地写成了 s['feats'].shape[0]，是一个复制/粘贴错误。
        # 正确的逻辑是直接获取 feats 列表中每个张量 s 的形状。
        sizes = [s.shape[0] for s in feats]
        labels = torch.tensor([s["target"] for s in samples], dtype=torch.long)
        
        target_size = max(sizes)
        collated_feats = torch.zeros(len(feats), target_size, feats[0].size(-1))
        padding_mask = torch.ones(len(feats), target_size).bool() # True for padded areas

        for i, (feat, size) in enumerate(zip(feats, sizes)):
            collated_feats[i, :size] = feat
            padding_mask[i, :size] = False # False for valid data
            
        return {
            "net_input": {"x": collated_feats, "padding_mask": padding_mask},
            "labels": labels
        }

# ==========================================================================================
# 核心功能函数
# ==========================================================================================
def load_model_weights(model, weights_path):
    """加载权重到模型"""
    logger.info(f"正在从 {weights_path} 加载权重...")
    try:
        # 修复: 将 weights_only=False 以兼容包含 numpy 对象的权重文件
        checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        
        # 检查是否是完整的训练检查点（包含 model_state_dict）
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            logger.info("检测到完整的训练检查点格式，提取 model_state_dict...")
            actual_state_dict = checkpoint["model_state_dict"]
            
            # 检查提取的 state_dict 是否直接兼容
            if "student_encoder.pre_net.weight" in actual_state_dict:
                # 检查是否包含teacher网络（完整SSRL模型）
                if "teacher_encoder.pre_net.weight" in actual_state_dict:
                    logger.info("检测到完整的SSRL模型（包含teacher和student），只提取student网络权重...")
                    # 只提取student网络的权重
                    student_state_dict = {}
                    for key, value in actual_state_dict.items():
                        if key.startswith('student_'):
                            student_state_dict[key] = value
                    
                    if not student_state_dict:
                        raise RuntimeError("未能从完整SSRL模型中提取到student网络权重。")
                    
                    model.load_state_dict(student_state_dict)
                    logger.info(f"成功从完整SSRL模型中提取并加载了 {len(student_state_dict)} 个student网络权重。")
                else:
                    # 直接兼容 SSRL 模型格式
                    model.load_state_dict(actual_state_dict)
                    logger.info("成功加载了完整的SSRL模型 state_dict。")
            elif "pre_net.weight" in actual_state_dict:
                # 需要重命名映射
                state_dict = model.state_dict()
                loaded_keys = []
                for key, value in actual_state_dict.items():
                    new_key = key
                    if key.startswith('pre_net'):
                        new_key = f"student_encoder.{key}"
                    elif key.startswith('post_net'):
                        new_key = f"student_classifier.{key.replace('post_net', 'fc_layer')}"
                    
                    if new_key in state_dict:
                        state_dict[new_key].copy_(value)
                        loaded_keys.append(new_key)
                    else:
                        logger.warning(f"权重key '{key}' (映射为 '{new_key}') 在模型中不存在，已跳过。")
                
                if not loaded_keys:
                    raise RuntimeError("未能从检查点中加载任何权重，请检查权重文件和模型结构。")
                    
                logger.info(f"成功从检查点加载并映射了 {len(loaded_keys)} 个权重。")
            else:
                raise RuntimeError(f"检查点中的 model_state_dict 格式无法识别。键包括: {list(actual_state_dict.keys())[:5]}...")
                
        # 如果不是检查点格式，按原来的逻辑处理        
        elif isinstance(checkpoint, dict) and "student_encoder" in str(list(checkpoint.keys())[0]):
             # 兼容从完整SSRL模型保存的权重 (e.g., fine-tuned model)
            model.load_state_dict(checkpoint)
            logger.info("成功加载了完整的SSRL模型 state_dict。")
        # 修复：不再依赖文件扩展名，而是通过检查key来判断是否需要重命名
        # 如果权重key是 'pre_net.weight' 这种格式，就需要手动映射
        elif isinstance(checkpoint, dict) and 'pre_net.weight' in checkpoint:
             # 兼容您的 emotion2vec_base.pt 或 train_for_clean_models/*.ckpt 预训练权重格式
            state_dict = model.state_dict()
            # pre_net -> student_encoder.pre_net
            # post_net -> student_classifier.fc_layer
            loaded_keys = []
            for key, value in checkpoint.items():
                new_key = key
                if key.startswith('pre_net'):
                    new_key = f"student_encoder.{key}"
                elif key.startswith('post_net'):
                    new_key = f"student_classifier.{key.replace('post_net', 'fc_layer')}"
                
                if new_key in state_dict:
                    state_dict[new_key].copy_(value)
                    loaded_keys.append(new_key)
                else:
                     logger.warning(f"权重key '{key}' (映射为 '{new_key}') 在模型中不存在，已跳过。")
            
            if not loaded_keys:
                raise RuntimeError("未能加载任何权重，请检查权重文件和模型结构。")
                
            logger.info(f"成功加载并映射了 {len(loaded_keys)} 个预训练权重。")
        else: # 兼容其他只包含 state_dict 的 .pth 文件
            model.load_state_dict(checkpoint)
            logger.info("成功加载了 state_dict。")

    except Exception as e:
        logger.error(f"加载权重失败: {e}")
        logger.error("请确保权重文件与模型结构匹配。")
        import traceback
        traceback.print_exc()
        return False
    return True

@torch.no_grad()
def extract_embeddings(model, dataloader):
    """使用模型从数据中提取所有嵌入和标签"""
    model.eval()
    model.to(DEVICE)
    
    all_embeddings = []
    all_labels = []
    
    logger.info("开始提取特征嵌入...")
    for batch in dataloader:
        net_input = batch["net_input"]
        feats = net_input["x"].to(DEVICE)
        padding_mask = net_input["padding_mask"].to(DEVICE)
        labels = batch["labels"]
        
        embeddings = model.get_embeddings(feats, padding_mask)
        
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
    logger.info("特征提取完成。")
    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)

def run_tsne(data, n_components=2, perplexity=30, n_iter=1000, random_state=42):
    """运行 t-SNE 降维"""
    logger.info(f"开始运行 t-SNE... (perplexity={perplexity}, n_iter={n_iter})")
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init='pca',
        learning_rate='auto'
    )
    tsne_results = tsne.fit_transform(data)
    logger.info("t-SNE 降维完成。")
    return tsne_results

def plot_tsne_comparison(tsne_results1, tsne_results2, labels, title_info, class_counts):
    """绘制 t-SNE 对比图，参考用户提供的风格"""
    # 设置图片大小和布局，参考用户的风格
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # 获取噪声等级信息
    db_info = title_info.upper() if title_info != "Unknown DB" else "Unknown"
    
    # 设置左图标题 - 初始权重
    ax1.set_title("(a).Initial Weights Encoder Features - All 5 Sessions", 
                  fontsize=14, fontweight='bold', pad=20)
    
    # 设置右图标题 - DAD训练后
    ax2.set_title(f"(b).After DAD Training (10 dB Noise Adapted) Encoder Features - All 5 Sessions", 
                  fontsize=14, fontweight='bold', pad=20)

    # 计算两个数据集的总体坐标范围，确保两个子图使用相同的坐标范围
    all_x = np.concatenate([tsne_results1[:, 0], tsne_results2[:, 0]])
    all_y = np.concatenate([tsne_results1[:, 1], tsne_results2[:, 1]])
    
    # 添加一些边距
    x_margin = (all_x.max() - all_x.min()) * 0.05
    y_margin = (all_y.max() - all_y.min()) * 0.05
    
    x_min, x_max = all_x.min() - x_margin, all_x.max() + x_margin
    y_min, y_max = all_y.min() - y_margin, all_y.max() + y_margin

    # 绘制散点图
    for class_name in CLASS_NAMES:
        class_idx = LABEL_DICT[class_name]
        indices = (labels == class_idx)
        color = EMOTION_COLORS[class_name]
        label = f"{EMOTION_LABELS[class_name]} (n={class_counts[class_name]})"
        
        # 左图：预训练模型
        ax1.scatter(tsne_results1[indices, 0], tsne_results1[indices, 1], 
                   c=color, label=label, alpha=0.7, s=20, edgecolors='none')
        
        # 右图：训练后模型
        ax2.scatter(tsne_results2[indices, 0], tsne_results2[indices, 1], 
                   c=color, label=label, alpha=0.7, s=20, edgecolors='none')

    # 设置坐标轴标签和属性
    for ax in [ax1, ax2]:
        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.legend(loc="upper right", fontsize=11, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 设置相同的坐标轴范围，确保两个子图尺寸一致
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')

    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    save_path = f"IEMOCAP_10dB_DAD_train_tsne.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"t-SNE 对比图已保存至: {save_path}")
    plt.show()
    
    return save_path

def calculate_cluster_metrics(embeddings, labels):
    """计算聚类评估指标"""
    try:
        silhouette = silhouette_score(embeddings, labels)
        calinski = calinski_harabasz_score(embeddings, labels)
        return silhouette, calinski
    except Exception as e:
        logger.warning(f"计算聚类指标时出错: {e}")
        return 0.0, 0.0

def generate_analysis_report(pretrained_embeddings, finetuned_embeddings, labels, class_counts, 
                           title_info, save_path):
    """生成详细的分析报告JSON文件"""
    
    # 计算聚类指标
    sil_initial, cal_initial = calculate_cluster_metrics(pretrained_embeddings, labels)
    sil_trained, cal_trained = calculate_cluster_metrics(finetuned_embeddings, labels)
    
    # 构建报告数据
    report = {
        "analysis_info": {
            "timestamp": datetime.now().isoformat(),
            "data_source": NOISY_DATA_ROOT_PATH,
            "initial_weights": PRETRAINED_WEIGHTS_PATH,
            "trained_weights": FINETUNED_WEIGHTS_PATH,
            "visualization_method": "t-SNE",
            "noise_level": title_info,
            "output_image": save_path
        },
        "data_statistics": {
            "initial_model_samples": len(labels),
            "trained_model_samples": len(labels),
            "total_samples": len(labels),
            "emotion_classes": [EMOTION_LABELS[name] for name in CLASS_NAMES],
            "class_distribution": {
                EMOTION_LABELS[name]: int(class_counts[name]) for name in CLASS_NAMES
            }
        },
        "cluster_analysis": {
            "initial_weights": {
                "silhouette_score": float(sil_initial),
                "calinski_harabasz_score": float(cal_initial)
            },
            "trained_weights": {
                "silhouette_score": float(sil_trained),
                "calinski_harabasz_score": float(cal_trained)
            },
            "improvement": {
                "silhouette_improvement": float(sil_trained - sil_initial),
                "calinski_improvement": float(cal_trained - cal_initial),
                "silhouette_percentage": float((sil_trained - sil_initial) / abs(sil_initial) * 100) if sil_initial != 0 else 0.0
            }
        },
        "model_analysis": {
            "feature_dimensions": {
                "initial_model": list(pretrained_embeddings.shape),
                "trained_model": list(finetuned_embeddings.shape)
            },
            "tsne_parameters": {
                "n_components": 2,
                "perplexity": 30,
                "n_iter": 1000,
                "random_state": 42
            }
        },
        "interpretation": {
            "purpose": f"Compare encoder feature representations before and after DAD training on {title_info} noise data",
            "data_scope": "All 5 sessions labeled noise data from IEMOCAP dataset", 
            "coverage": "Complete IEMOCAP dataset across all sessions",
            "expected_outcome": "Better class separation and domain adaptation after DAD training",
            "metrics_explanation": {
                "silhouette_score": "Range [-1, 1], higher is better. Measures how well samples are clustered within their own class vs. other classes.",
                "calinski_harabasz_score": "Higher is better. Ratio of between-cluster to within-cluster variance."
            },
            "results_summary": {
                "clustering_improved": bool(sil_trained > sil_initial),
                "separation_improved": bool(cal_trained > cal_initial),
                "overall_assessment": "Improved" if (sil_trained > sil_initial and cal_trained > cal_initial) else "Mixed" if (sil_trained > sil_initial or cal_trained > cal_initial) else "Degraded"
            }
        }
    }
    
    # 保存报告
    report_path = f"tsne_analysis_report_{title_info.replace(' ', '_')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    logger.info(f"分析报告已保存至: {report_path}")
    
    # 打印关键结果
    logger.info("=== 聚类分析结果 ===")
    logger.info(f"初始模型 - Silhouette Score: {sil_initial:.4f}, Calinski-Harabasz Score: {cal_initial:.2f}")
    logger.info(f"训练后模型 - Silhouette Score: {sil_trained:.4f}, Calinski-Harabasz Score: {cal_trained:.2f}")
    logger.info(f"改进情况: Silhouette {sil_trained-sil_initial:+.4f}, Calinski-Harabasz {cal_trained-cal_initial:+.2f}")
    
    return report_path

# ==========================================================================================
# 主执行函数
# ==========================================================================================
def main():
    logger.info("====== 开始 t-SNE 可视化脚本 ======")
    
    # --- 1. 提取dB信息 ---
    db_match = re.search(r'(\d+db)', FINETUNED_WEIGHTS_PATH, re.IGNORECASE)
    title_info = db_match.group(1) if db_match else "Unknown DB"
    logger.info(f"从路径中提取的噪声等级为: {title_info}")

    # --- 2. 加载数据 ---
    try:
        dataset = IEMOCAP_tSNE_Dataset(NOISY_DATA_ROOT_PATH)
        # 使用较大的 batch_size 以加快特征提取速度
        dataloader = DataLoader(dataset, batch_size=128, collate_fn=dataset.collator, shuffle=False)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"数据加载失败: {e}")
        logger.error("请检查 NOISY_DATA_ROOT_PATH 是否正确，并确保该目录下有 train.npy, train.lengths, 和 train.emo 文件。")
        return

    # --- 3. 处理预训练模型 ---
    logger.info("\n--- 处理预训练模型 ---")
    pretrained_model = SSRLModel(input_dim=768, hidden_dim=256, num_classes=len(CLASS_NAMES))
    if not load_model_weights(pretrained_model, PRETRAINED_WEIGHTS_PATH):
        return
    pretrained_embeddings, labels = extract_embeddings(pretrained_model, dataloader)
    
    # --- 4. 处理主干训练后模型 ---
    logger.info("\n--- 处理主干训练后模型 ---")
    finetuned_model = SSRLModel(input_dim=768, hidden_dim=256, num_classes=len(CLASS_NAMES))
    if not load_model_weights(finetuned_model, FINETUNED_WEIGHTS_PATH):
        return
    finetuned_embeddings, _ = extract_embeddings(finetuned_model, dataloader)
    
    # --- 5. 运行 t-SNE ---
    logger.info("\n--- 运行 t-SNE 降维 (可能需要一些时间) ---")
    tsne_pretrained = run_tsne(pretrained_embeddings)
    tsne_finetuned = run_tsne(finetuned_embeddings)
    
    # --- 6. 绘图 ---
    logger.info("\n--- 绘制对比图 ---")
    save_path = plot_tsne_comparison(tsne_pretrained, tsne_finetuned, labels, title_info, dataset.class_counts)
    
    # --- 7. 生成分析报告 ---
    logger.info("\n--- 生成分析报告 ---")
    report_path = generate_analysis_report(pretrained_embeddings, finetuned_embeddings, labels, 
                                         dataset.class_counts, title_info, save_path)

    logger.info("\n====== t-SNE 可视化脚本执行完毕 ======")
    logger.info(f"输出文件:")
    logger.info(f"  - 可视化图片: {save_path}")
    logger.info(f"  - 分析报告: {report_path}")

if __name__ == "__main__":
    main() 