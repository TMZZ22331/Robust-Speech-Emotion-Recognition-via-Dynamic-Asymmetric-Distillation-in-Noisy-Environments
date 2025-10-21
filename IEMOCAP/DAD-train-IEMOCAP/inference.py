#!/usr/bin/env python3
"""
IEMOCAP跨域推理脚本
加载训练好的权重，在不同噪声环境下进行跨域测试
支持自定义权重路径和测试数据路径
"""

# ===========================================
# 🔧 用户自定义配置区域
# ===========================================

# 权重文件路径配置
WEIGHT_CONFIG = {
    "model_path": r"C:\Users\admin\Desktop\111\iemocap_cross_domain_results\20db\fold_4\models\iemocap_cross_domain_best.pth",  # 训练好的模型权重路径
    "description": "SNR=20dB IEMOCAP 噪声环境下训练的模型"  # 权重描述
}

# 测试数据配置
TEST_DATA_CONFIG = {
    "test_data_path": r"C:\Users\admin\Desktop\DATA\fix_CASIA\processed_features_noisy_20db\train",  # 测试数据路径（去掉扩展名）
    "noise_description": "CASIA_20db",  # 噪声描述（用于结果命名）
    "fold_id": 3  # 使用哪个fold的数据划分
}

# 推理配置
INFERENCE_CONFIG = {
    "batch_size": 32,  # 推理批次大小
    "save_results": True,  # 是否保存详细结果
    "generate_plots": True,  # 是否生成可视化图表
    "results_base_dir": "cross_domain_inference_results"  # 结果保存基础目录
}

# ===========================================
# 📦 导入和基础设置
# ===========================================

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, balanced_accuracy_score, precision_recall_fscore_support
)
import json
from datetime import datetime
import re
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入必要模块
import config as cfg
from model import SSRLModel
from dataload_noisy import get_cv_dataloaders_noisy

# 导入CASIA数据加载器
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../CASIA/DAD-train-CASIA'))
from dataload_casia_noisy import create_casia_noisy_dataloaders_with_speaker_isolation

class IEMOCAPCrossDomainInference:
    """
    IEMOCAP跨域推理器
    加载训练好的权重，在新的噪声环境下进行测试
    """
    
    def __init__(self, weight_config, test_data_config, inference_config):
        """初始化推理器"""
        self.weight_config = weight_config
        self.test_data_config = test_data_config
        self.inference_config = inference_config
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() and cfg.USE_CUDA else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 提取权重信息
        self.source_model_info = self._extract_model_info()
        
        # 设置结果目录
        self.results_dir = self._setup_results_directory()
        
        logger.info("="*60)
        logger.info("🚀 IEMOCAP跨域推理配置")
        logger.info("="*60)
        logger.info(f"📁 权重路径: {self.weight_config['model_path']}")
        logger.info(f"📄 权重描述: {self.weight_config['description']}")
        logger.info(f"🗂️ 测试数据: {self.test_data_config['test_data_path']}")
        logger.info(f"🔊 噪声环境: {self.test_data_config['noise_description']}")
        logger.info(f"📊 批次大小: {self.inference_config['batch_size']}")
        logger.info(f"🎯 推理网络: Student Network (学生网络)")
        logger.info(f"💾 结果目录: {self.results_dir}")
        logger.info("="*60)
        
        # 初始化组件
        self._setup_components()
    
    def _extract_model_info(self):
        """从权重路径中提取模型训练信息"""
        model_path = self.weight_config['model_path']
        
        # 尝试提取噪声等级信息
        source_noise = "unknown"
        fold_info = "unknown"
        
        # 提取噪声等级
        noise_patterns = [r'(\d+db)', r'(\d+)db', r'noisy_(\d+)db']
        for pattern in noise_patterns:
            match = re.search(pattern, model_path.lower())
            if match:
                source_noise = f"{match.group(1)}"
                break
        
        # 提取fold信息
        fold_match = re.search(r'fold_(\d+)', model_path.lower())
        if fold_match:
            fold_info = f"fold_{fold_match.group(1)}"
        
        return {
            'source_noise': source_noise,
            'fold': fold_info,
            'full_path': model_path
        }
    
    def _setup_results_directory(self):
        """设置结果保存目录"""
        base_dir = self.inference_config['results_base_dir']
        
        # 创建目录名称
        source_info = f"{self.source_model_info['source_noise']}_{self.source_model_info['fold']}"
        target_info = f"test_on_{self.test_data_config['noise_description']}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_dir = os.path.join(base_dir, f"{source_info}_to_{target_info}_{timestamp}")
        
        # 创建目录结构
        os.makedirs(results_dir, exist_ok=True)
        if self.inference_config['save_results']:
            os.makedirs(os.path.join(results_dir, "reports"), exist_ok=True)
        if self.inference_config['generate_plots']:
            os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
        
        return results_dir
    
    def _setup_components(self):
        """设置推理组件"""
        self._setup_model()
        self._load_weights()
        self._setup_dataloader()
    
    def _setup_model(self):
        """设置模型"""
        logger.info("🏗️ 初始化SSRL模型...")
        self.model = SSRLModel(cfg=cfg).to(self.device)
        
        if cfg.PRINT_MODEL_INFO:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"📊 模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
        
        logger.info("✅ SSRL模型初始化完成")
    
    def _load_weights(self):
        """加载预训练权重"""
        model_path = self.weight_config['model_path']
        
        logger.info(f"💾 加载权重: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"权重文件不存在: {model_path}")
        
        try:
            # 加载checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 提取模型状态字典
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                epoch_info = checkpoint.get('epoch', 'unknown')
                logger.info(f"📈 权重来源: Epoch {epoch_info + 1}" if isinstance(epoch_info, int) else f"📈 权重来源: {epoch_info}")
                
                # 显示训练时的性能信息
                if 'clean_results' in checkpoint and 'noisy_results' in checkpoint:
                    clean_acc = checkpoint['clean_results'].get('weighted_accuracy', 'N/A')
                    noisy_acc = checkpoint['noisy_results'].get('weighted_accuracy', 'N/A')
                    logger.info(f"🎯 训练时性能 - Clean: {clean_acc:.2f}%, Noisy: {noisy_acc:.2f}%")
            else:
                state_dict = checkpoint
                logger.info("📈 权重来源: 直接状态字典")
            
            # 加载权重到模型
            self.model.load_state_dict(state_dict)
            logger.info("✅ 权重加载成功")
            
        except Exception as e:
            logger.error(f"❌ 权重加载失败: {e}")
            raise
    
    def _detect_dataset_type(self, data_path):
        """检测数据集类型"""
        # 检查文件扩展名来判断数据集类型
        if os.path.exists(f"{data_path}.lbl") and os.path.exists(f"{data_path}.spk"):
            return "CASIA"
        elif os.path.exists(f"{data_path}.emo"):
            return "IEMOCAP"
        else:
            # 尝试根据路径名判断
            path_lower = data_path.lower()
            if 'casia' in path_lower:
                return "CASIA"
            elif 'iemocap' in path_lower:
                return "IEMOCAP"
            else:
                logger.warning("⚠️ 无法确定数据集类型，默认使用IEMOCAP加载器")
                return "IEMOCAP"

    def _setup_dataloader(self):
        """设置数据加载器"""
        logger.info("🗂️ 设置测试数据加载器...")
        
        test_data_path = self.test_data_config['test_data_path']
        batch_size = self.inference_config['batch_size']
        fold_id = self.test_data_config['fold_id']
        
        try:
            # 检测数据集类型
            dataset_type = self._detect_dataset_type(test_data_path)
            logger.info(f"🔍 检测到数据集类型: {dataset_type}")
            
            if dataset_type == "CASIA":
                # 使用CASIA数据加载器
                logger.info("🔊 使用CASIA数据加载器进行推理")
                
                # CASIA使用0-3的fold，需要转换
                casia_fold = fold_id - 1 if fold_id > 0 else 0
                casia_fold = max(0, min(3, casia_fold))  # 确保在0-3范围内
                
                logger.info(f"   📋 IEMOCAP fold {fold_id} → CASIA fold {casia_fold}")
                
                # 创建CASIA数据加载器，只使用验证集进行推理
                _, _, self.test_loader = create_casia_noisy_dataloaders_with_speaker_isolation(
                    test_data_path, batch_size, fold=casia_fold
                )
                
                # CASIA类别信息
                self.class_names = ['angry', 'happy', 'neutral', 'sad']
                self.num_classes = 4
                
            else:
                # 使用IEMOCAP数据加载器
                logger.info("🔊 使用IEMOCAP数据加载器进行推理")
                _, _, self.test_loader = get_cv_dataloaders_noisy(
                    test_data_path, batch_size, fold_id=fold_id
                )
                
                # 获取类别信息（从配置中获取或使用默认值）
                if hasattr(cfg, 'CLASS_NAMES'):
                    self.class_names = cfg.CLASS_NAMES
                    self.num_classes = len(cfg.CLASS_NAMES)
                elif hasattr(cfg, 'NUM_CLASSES'):
                    self.num_classes = cfg.NUM_CLASSES
                    self.class_names = [f"Class_{i}" for i in range(self.num_classes)]
                else:
                    # 默认情况，假设是情感分类任务
                    self.num_classes = 4
                    self.class_names = ['angry', 'happy', 'neutral', 'sad']
                    logger.warning("⚠️ 未找到类别配置，使用默认4类情感分类")
            
            logger.info(f"✅ 测试数据加载器设置完成")
            logger.info(f"   📊 测试批次数: {len(self.test_loader)}")
            logger.info(f"   🏷️ 类别数量: {self.num_classes}")
            logger.info(f"   📝 类别名称: {self.class_names}")
            
        except Exception as e:
            logger.error(f"❌ 数据加载器设置失败: {e}")
            raise
    

    
    def run_inference(self):
        """运行推理"""
        logger.info("🚀 开始跨域推理...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                # 提取数据
                feats = batch['net_input']['feats'].to(self.device)
                padding_mask = batch['net_input']['padding_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 使用学生网络进行推理（推荐方式）
                outputs = self.model.predict(feats, padding_mask, use_teacher=False)
                
                # 获取预测结果
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # 收集结果
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # 显示进度
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  处理进度: {batch_idx + 1}/{len(self.test_loader)} 批次")
        
        logger.info("✅ 推理完成，开始计算评估指标...")
        
        # 计算评估指标
        results = self._calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        # 保存结果
        if self.inference_config['save_results']:
            self._save_results(results, all_labels, all_predictions)
        
        # 生成可视化
        if self.inference_config['generate_plots']:
            self._generate_plots(results)
        
        return results
    
    def _calculate_metrics(self, true_labels, predictions, probabilities):
        """计算详细的评估指标"""
        
        # 基本指标
        accuracy = accuracy_score(true_labels, predictions) * 100
        weighted_accuracy = balanced_accuracy_score(true_labels, predictions) * 100
        f1_weighted = f1_score(true_labels, predictions, average='weighted', zero_division=0) * 100
        f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0) * 100
        
        # 混淆矩阵
        cm = confusion_matrix(true_labels, predictions, labels=range(self.num_classes))
        
        # 每类指标
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0, labels=range(self.num_classes)
        )
        
        # 每类准确率
        per_class_accuracy = []
        for i in range(self.num_classes):
            if cm[i, :].sum() > 0:
                per_class_accuracy.append(cm[i, i] / cm[i, :].sum())
            else:
                per_class_accuracy.append(0.0)
        
        # 置信度统计
        probabilities = np.array(probabilities)
        max_probs = np.max(probabilities, axis=1)
        confidence_stats = {
            'mean_confidence': float(np.mean(max_probs)),
            'std_confidence': float(np.std(max_probs)),
            'min_confidence': float(np.min(max_probs)),
            'max_confidence': float(np.max(max_probs))
        }
        
        results = {
            'overview': {
                'accuracy': accuracy,
                'weighted_accuracy': weighted_accuracy,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'total_samples': len(true_labels)
            },
            'per_class': {
                'class_names': self.class_names,
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1_score': f1.tolist(),
                'accuracy': per_class_accuracy,
                'support': support.tolist()
            },
            'confusion_matrix': cm.tolist(),
            'confidence_stats': confidence_stats,
            'test_info': {
                'source_model': self.source_model_info,
                'test_data': self.test_data_config,
                'inference_config': self.inference_config
            }
        }
        
        return results
    
    def _save_results(self, results, true_labels, predictions):
        """保存详细结果到JSON文件"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建完整报告
        full_report = {
            'experiment_info': {
                'timestamp': timestamp,
                'source_model_path': self.weight_config['model_path'],
                'source_model_description': self.weight_config['description'],
                'test_data_path': self.test_data_config['test_data_path'],
                'test_noise_description': self.test_data_config['noise_description'],
                'cross_domain_type': f"{self.source_model_info['source_noise']} → {self.test_data_config['noise_description']}"
            },
            'results': results,
            'detailed_classification_report': classification_report(
                true_labels, predictions, target_names=self.class_names, output_dict=True, zero_division=0
            )
        }
        
        # 保存完整报告
        report_path = os.path.join(self.results_dir, "reports", f"cross_domain_inference_report_{timestamp}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=4, ensure_ascii=False)
        
        logger.info(f"📄 详细报告已保存: {report_path}")
        
        # 保存简化版本用于快速查看
        summary_report = {
            'cross_domain_test': f"{self.source_model_info['source_noise']} → {self.test_data_config['noise_description']}",
            'performance': {
                'accuracy': f"{results['overview']['accuracy']:.2f}%",
                'weighted_accuracy': f"{results['overview']['weighted_accuracy']:.2f}%",
                'weighted_f1': f"{results['overview']['f1_weighted']:.2f}%",
                'macro_f1': f"{results['overview']['f1_macro']:.2f}%"
            },
            'confidence': {
                'mean': f"{results['confidence_stats']['mean_confidence']:.4f}",
                'std': f"{results['confidence_stats']['std_confidence']:.4f}"
            }
        }
        
        summary_path = os.path.join(self.results_dir, "reports", f"quick_summary_{timestamp}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=4, ensure_ascii=False)
        
        logger.info(f"📋 快速摘要已保存: {summary_path}")
    
    def _generate_plots(self, results):
        """生成可视化图表"""
        
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 混淆矩阵
        plt.figure(figsize=(12, 10))
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        
        title = f'Cross-Domain Confusion Matrix\n{self.source_model_info["source_noise"]} → {self.test_data_config["noise_description"]}'
        plt.title(title, fontsize=14, weight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # 添加性能信息
        acc_text = f'Accuracy: {results["overview"]["accuracy"]:.2f}%\nWeighted Acc: {results["overview"]["weighted_accuracy"]:.2f}%'
        plt.figtext(0.02, 0.02, acc_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        cm_path = os.path.join(self.results_dir, "plots", f"confusion_matrix_{timestamp}.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 混淆矩阵已保存: {cm_path}")
        
        # 2. 每类性能柱状图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        x = np.arange(len(self.class_names))
        
        # Precision
        ax1.bar(x, results['per_class']['precision'], color='skyblue', alpha=0.7)
        ax1.set_title('Precision per Class', fontweight='bold')
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Precision')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Recall
        ax2.bar(x, results['per_class']['recall'], color='lightcoral', alpha=0.7)
        ax2.set_title('Recall per Class', fontweight='bold')
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Recall')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        # F1-Score
        ax3.bar(x, results['per_class']['f1_score'], color='lightgreen', alpha=0.7)
        ax3.set_title('F1-Score per Class', fontweight='bold')
        ax3.set_xlabel('Classes')
        ax3.set_ylabel('F1-Score')
        ax3.set_xticks(x)
        ax3.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        # Accuracy
        ax4.bar(x, results['per_class']['accuracy'], color='gold', alpha=0.7)
        ax4.set_title('Accuracy per Class', fontweight='bold')
        ax4.set_xlabel('Classes')
        ax4.set_ylabel('Accuracy')
        ax4.set_xticks(x)
        ax4.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'Per-Class Performance\n{self.source_model_info["source_noise"]} → {self.test_data_config["noise_description"]}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        metrics_path = os.path.join(self.results_dir, "plots", f"per_class_metrics_{timestamp}.png")
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 每类性能图表已保存: {metrics_path}")
    
    def print_results_summary(self, results):
        """打印结果摘要"""
        
        logger.info("\n" + "="*60)
        logger.info("🎯 跨域推理结果摘要")
        logger.info("="*60)
        
        logger.info(f"🔄 跨域类型: {self.source_model_info['source_noise']} → {self.test_data_config['noise_description']}")
        logger.info(f"📊 测试样本数: {results['overview']['total_samples']}")
        logger.info(f"🎯 准确率: {results['overview']['accuracy']:.2f}%")
        logger.info(f"⚖️ 加权准确率: {results['overview']['weighted_accuracy']:.2f}%")
        logger.info(f"🏆 加权F1分数: {results['overview']['f1_weighted']:.2f}%")
        logger.info(f"📈 宏平均F1分数: {results['overview']['f1_macro']:.2f}%")
        
        logger.info(f"\n🔍 置信度统计:")
        logger.info(f"   平均置信度: {results['confidence_stats']['mean_confidence']:.4f}")
        logger.info(f"   置信度标准差: {results['confidence_stats']['std_confidence']:.4f}")
        logger.info(f"   最低置信度: {results['confidence_stats']['min_confidence']:.4f}")
        logger.info(f"   最高置信度: {results['confidence_stats']['max_confidence']:.4f}")
        
        logger.info(f"\n📋 每类性能详情:")
        for i, class_name in enumerate(self.class_names):
            logger.info(f"   {class_name}: "
                       f"Prec={results['per_class']['precision'][i]:.3f}, "
                       f"Rec={results['per_class']['recall'][i]:.3f}, "
                       f"F1={results['per_class']['f1_score'][i]:.3f}, "
                       f"Acc={results['per_class']['accuracy'][i]:.3f}, "
                       f"Support={results['per_class']['support'][i]}")
        
        logger.info("="*60)

def main():
    """主函数"""
    
    logger.info("🚀 启动IEMOCAP跨域推理脚本")
    
    # 验证配置
    if not os.path.exists(WEIGHT_CONFIG["model_path"]):
        logger.error(f"❌ 权重文件不存在: {WEIGHT_CONFIG['model_path']}")
        logger.error("请检查WEIGHT_CONFIG中的model_path配置")
        return
    
    # 验证测试数据路径
    test_data_path = TEST_DATA_CONFIG["test_data_path"]
    data_exists = False
    
    # 检查CASIA格式数据文件
    if os.path.exists(f"{test_data_path}.npy") and os.path.exists(f"{test_data_path}.lbl"):
        data_exists = True
        logger.info(f"✅ 检测到CASIA格式数据文件")
    # 检查IEMOCAP格式数据文件
    elif os.path.exists(f"{test_data_path}.emo"):
        data_exists = True
        logger.info(f"✅ 检测到IEMOCAP格式数据文件")
    # 检查目录是否存在
    elif os.path.exists(test_data_path):
        data_exists = True
        logger.info(f"✅ 检测到数据目录")
    
    if not data_exists:
        logger.error(f"❌ 测试数据不存在: {test_data_path}")
        logger.error("请检查TEST_DATA_CONFIG中的test_data_path配置")
        logger.error("确保存在以下文件之一:")
        logger.error(f"  - CASIA格式: {test_data_path}.npy, {test_data_path}.lbl")
        logger.error(f"  - IEMOCAP格式: {test_data_path}.emo")
        logger.error(f"  - 或者目录存在: {test_data_path}")
        return
    
    try:
        # 创建推理器
        inferencer = IEMOCAPCrossDomainInference(
            weight_config=WEIGHT_CONFIG,
            test_data_config=TEST_DATA_CONFIG,
            inference_config=INFERENCE_CONFIG
        )
        
        # 运行推理
        results = inferencer.run_inference()
        
        # 打印结果摘要
        inferencer.print_results_summary(results)
        
        logger.info(f"\n✅ 跨域推理完成! 结果已保存至: {inferencer.results_dir}")
        
    except Exception as e:
        logger.error(f"❌ 推理过程中发生错误: {e}", exc_info=True)
        return

if __name__ == "__main__":
    main() 