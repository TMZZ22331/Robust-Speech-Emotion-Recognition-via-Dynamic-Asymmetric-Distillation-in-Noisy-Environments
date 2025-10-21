#!/usr/bin/env python3
"""
DACP和ECDA动态机制演化分析脚本
分析训练过程中DACP的动态阈值、类别质量分数以及ECDA的类别注意力权重的演变
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
from pathlib import Path

def load_training_history(history_filepath):
    """加载训练历史数据"""
    if not os.path.exists(history_filepath):
        raise FileNotFoundError(f"训练历史文件不存在: {history_filepath}")
    
    with open(history_filepath, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    print(f"✅ 成功加载训练历史数据: {history_filepath}")
    return history

def prepare_dataframes(history, class_names, warmup_epochs):
    """准备分析用的数据框"""
    # 检查必要的数据是否存在
    required_keys = ['dacp_ema_thresholds', 'dacp_class_quality', 'ecda_class_attention']
    missing_keys = [key for key in required_keys if key not in history]
    
    if missing_keys:
        raise ValueError(f"训练历史中缺少以下必要数据: {missing_keys}")
    
    # 创建epoch索引（从warmup结束后开始）
    num_epochs = len(history['dacp_ema_thresholds'])
    epochs = list(range(warmup_epochs + 1, warmup_epochs + 1 + num_epochs))
    
    # 转换为DataFrame
    df_thresholds = pd.DataFrame(
        history['dacp_ema_thresholds'], 
        columns=class_names, 
        index=epochs
    )
    
    df_quality = pd.DataFrame(
        history['dacp_class_quality'], 
        columns=class_names, 
        index=epochs
    )
    
    df_attention = pd.DataFrame(
        history['ecda_class_attention'], 
        columns=class_names, 
        index=epochs
    )
    
    print(f"📊 数据准备完成:")
    print(f"   - 分析轮次范围: Epoch {epochs[0]} - {epochs[-1]}")
    print(f"   - 情绪类别: {class_names}")
    print(f"   - 数据点数量: {num_epochs} 个epoch")
    
    return df_thresholds, df_quality, df_attention, epochs

def create_evolution_plots(df_thresholds, df_quality, df_attention, output_dir):
    """创建演化过程可视化图表"""
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # 创建主图表
    fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=True)
    fig.suptitle('DACP & ECDA Dynamic Mechanisms Evolution Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 子图1: DACP演化阈值
    for class_name in df_thresholds.columns:
        axes[0].plot(df_thresholds.index, df_thresholds[class_name], 
                    marker='o', markersize=4, linewidth=2.5, 
                    label=f'{class_name.upper()}', alpha=0.85)
    
    axes[0].set_title('DACP Evolving Thresholds (tau_c^t)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Threshold Value', fontsize=12)
    axes[0].legend(title='Emotion Class', loc='best', frameon=True, fancybox=True, shadow=True)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)
    
    # 添加防火墙线（阈值>1的区域）
    axes[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, 
                   linewidth=2, label='Firewall Threshold (τ=1)')
    axes[0].fill_between(df_thresholds.index, 1.0, axes[0].get_ylim()[1], 
                        color='red', alpha=0.1, label='Firewall Zone')
    
    # 子图2: DACP类别质量分数
    for class_name in df_quality.columns:
        axes[1].plot(df_quality.index, df_quality[class_name], 
                    marker='s', markersize=4, linewidth=2.5, 
                    label=f'{class_name.upper()}', alpha=0.85)
    
    axes[1].set_title('DACP Class Quality Scores (Q_c^e)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Quality Score', fontsize=12)
    axes[1].legend(title='Emotion Class', loc='best', frameon=True, fancybox=True, shadow=True)
    axes[1].grid(True, alpha=0.3)
    
    # 子图3: ECDA类别注意力权重
    for class_name in df_attention.columns:
        axes[2].plot(df_attention.index, df_attention[class_name], 
                    marker='^', markersize=4, linewidth=2.5, 
                    label=f'{class_name.upper()}', alpha=0.85)
    
    axes[2].set_title('ECDA Class Attention Weights (w_c^class)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Training Epoch', fontsize=12)
    axes[2].set_ylabel('Attention Weight', fontsize=12)
    axes[2].legend(title='Emotion Class', loc='best', frameon=True, fancybox=True, shadow=True)
    axes[2].grid(True, alpha=0.3)
    
    # 添加基线参考线
    axes[2].axhline(y=1.0, color='gray', linestyle=':', alpha=0.8, 
                   linewidth=2, label='Baseline (Weight=1)')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # 保存主图表
    main_plot_path = os.path.join(output_dir, 'DACP和ECDA机制演化图.png')
    plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ 主要演化分析图已保存至: {main_plot_path}")
    plt.close()
    
    return main_plot_path

def create_summary_statistics(df_thresholds, df_quality, df_attention, output_dir):
    """创建统计摘要图表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Dynamic Mechanisms Summary Statistics', fontsize=16, fontweight='bold')
    
    # 阈值统计
    threshold_stats = df_thresholds.describe()
    sns.heatmap(threshold_stats, annot=True, fmt='.3f', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('DACP Thresholds Statistics')
    
    # 质量分数统计
    quality_stats = df_quality.describe()
    sns.heatmap(quality_stats, annot=True, fmt='.3f', cmap='Greens', ax=axes[0,1])
    axes[0,1].set_title('Class Quality Statistics')
    
    # 注意力权重统计
    attention_stats = df_attention.describe()
    sns.heatmap(attention_stats, annot=True, fmt='.3f', cmap='Oranges', ax=axes[1,0])
    axes[1,0].set_title('Attention Weights Statistics')
    
    # 防火墙激活频率
    firewall_activation = (df_thresholds > 1.0).sum()
    axes[1,1].bar(firewall_activation.index, firewall_activation.values, 
                  color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    axes[1,1].set_title('Firewall Activation Frequency')
    axes[1,1].set_ylabel('Times Activated (τ > 1)')
    axes[1,1].set_xlabel('Emotion Class')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    stats_plot_path = os.path.join(output_dir, '动态机制统计摘要图.png')
    plt.savefig(stats_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ 统计摘要图已保存至: {stats_plot_path}")
    plt.close()
    
    return stats_plot_path

def generate_analysis_report(df_thresholds, df_quality, df_attention, output_dir):
    """生成分析报告"""
    report_data = {
        'analysis_summary': {
            'total_epochs_analyzed': len(df_thresholds),
            'emotion_classes': df_thresholds.columns.tolist(),
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'dacp_thresholds': {
            'mean_per_class': df_thresholds.mean().to_dict(),
            'std_per_class': df_thresholds.std().to_dict(),
            'max_per_class': df_thresholds.max().to_dict(),
            'firewall_activations': (df_thresholds > 1.0).sum().to_dict(),
            'firewall_rate': ((df_thresholds > 1.0).sum() / len(df_thresholds)).to_dict()
        },
        'class_quality_scores': {
            'mean_per_class': df_quality.mean().to_dict(),
            'std_per_class': df_quality.std().to_dict(),
            'trend_slope': {}  # 可以添加趋势分析
        },
        'ecda_attention_weights': {
            'mean_per_class': df_attention.mean().to_dict(),
            'std_per_class': df_attention.std().to_dict(),
            'above_baseline_rate': ((df_attention > 1.0).sum() / len(df_attention)).to_dict()
        }
    }
    
    # 计算趋势斜率
    for class_name in df_quality.columns:
        x = np.arange(len(df_quality))
        y = df_quality[class_name].values
        slope = np.polyfit(x, y, 1)[0]
        report_data['class_quality_scores']['trend_slope'][class_name] = float(slope)
    
    # 保存报告
    report_path = os.path.join(output_dir, 'DACP和ECDA机制分析报告.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)
    
    print(f"✅ 详细分析报告已保存至: {report_path}")
    return report_path

def analyze_dynamic_process(history_filepath, output_dir, class_names=None, warmup_epochs=30):
    """
    主分析函数：加载训练历史数据并生成完整的可视化分析
    """
    print("🔬 开始DACP和ECDA动态机制演化分析...")
    
    # 默认IEMOCAP情绪类别
    if class_names is None:
        class_names = ['angry', 'happy', 'neutral', 'sad']
    
    # 创建输出目录和分析子目录
    analysis_subdir = os.path.join(output_dir, "01_DACP和ECDA动态机制演化分析")
    os.makedirs(analysis_subdir, exist_ok=True)
    output_dir = analysis_subdir  # 重定向输出目录
    
    try:
        # 1. 加载数据
        history = load_training_history(history_filepath)
        
        # 2. 准备数据框
        df_thresholds, df_quality, df_attention, epochs = prepare_dataframes(
            history, class_names, warmup_epochs
        )
        
        # 3. 创建可视化图表
        main_plot = create_evolution_plots(df_thresholds, df_quality, df_attention, output_dir)
        stats_plot = create_summary_statistics(df_thresholds, df_quality, df_attention, output_dir)
        
        # 4. 生成分析报告
        report_path = generate_analysis_report(df_thresholds, df_quality, df_attention, output_dir)
        
        print("\n🎉 DACP和ECDA动态机制分析完成!")
        print(f"📊 生成的文件:")
        print(f"   - 主要演化图: {main_plot}")
        print(f"   - 统计摘要图: {stats_plot}")
        print(f"   - 分析报告: {report_path}")
        
        return {
            'main_plot': main_plot,
            'stats_plot': stats_plot,
            'report': report_path
        }
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='分析DACP和ECDA动态机制的演化过程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    python analyze_dacp_evolution.py --path /path/to/training_history.json
    python analyze_dacp_evolution.py --path /path/to/training_history.json --output ./analysis_results
    python analyze_dacp_evolution.py --path /path/to/training_history.json --classes ang hap neu sad --warmup 30
        """
    )
    
    parser.add_argument('--path', type=str, required=True,
                       help='训练历史JSON文件的路径 (training_history.json)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出目录路径 (默认为历史文件所在目录)')
    parser.add_argument('--classes', nargs='+', default=['angry', 'happy', 'neutral', 'sad'],
                       help='情绪类别名称列表 (默认: angry happy neutral sad)')
    parser.add_argument('--warmup', type=int, default=30,
                       help='预热轮次数 (默认: 30)')
    
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output is None:
        output_directory = os.path.dirname(args.path)
    else:
        output_directory = args.output
    
    # 执行分析
    try:
        results = analyze_dynamic_process(
            history_filepath=args.path,
            output_dir=output_directory,
            class_names=args.classes,
            warmup_epochs=args.warmup
        )
        print(f"\n✨ 分析完成! 结果已保存至: {output_directory}")
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 