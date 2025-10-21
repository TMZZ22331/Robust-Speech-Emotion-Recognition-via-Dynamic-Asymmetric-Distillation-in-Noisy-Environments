#!/usr/bin/env python3
"""
确认偏差分析脚本
通过追踪固定样本在训练过程中的伪标签变化来量化确认偏差
分析DACP防火墙机制与标签稳定性的关系
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from collections import defaultdict, Counter
from scipy import stats

def load_analysis_data(log_filepath, history_filepath):
    """加载确认偏差日志和训练历史数据"""
    print("📂 加载分析数据...")
    
    # 加载确认偏差日志
    if not os.path.exists(log_filepath):
        raise FileNotFoundError(f"确认偏差日志文件不存在: {log_filepath}")
    
    with open(log_filepath, 'r', encoding='utf-8') as f:
        log_data = json.load(f)
    
    df_log = pd.DataFrame(log_data)
    print(f"✅ 确认偏差日志加载完成，共 {len(df_log)} 条记录")
    
    # 加载训练历史（可选）
    history = None
    if history_filepath and os.path.exists(history_filepath):
        with open(history_filepath, 'r', encoding='utf-8') as f:
            history = json.load(f)
        print("✅ 训练历史数据加载完成")
    else:
        print("⚠️ 训练历史文件不存在，将跳过相关分析")
    
    return df_log, history

def analyze_label_consistency(df_log, class_names=None):
    """分析伪标签的一致性和翻转情况"""
    print("🔍 分析伪标签一致性...")
    
    if class_names is None:
        class_names = ['angry', 'happy', 'neutral', 'sad']
    
    # 创建样本-epoch透视表
    df_pivot = df_log.pivot_table(
        index='sample_id', 
        columns='epoch', 
        values='pseudo_label', 
        aggfunc='first'
    )
    
    # 计算翻转统计
    total_samples = len(df_pivot)
    total_epochs = len(df_pivot.columns)
    
    # 计算每个样本的翻转次数
    flips_per_sample = df_pivot.diff(axis=1).ne(0).sum(axis=1)
    
    # 计算翻转率
    flip_rates = flips_per_sample / (total_epochs - 1)
    
    # 统计信息
    consistency_stats = {
        'total_samples_tracked': total_samples,
        'total_epochs': total_epochs,
        'mean_flips_per_sample': flips_per_sample.mean(),
        'std_flips_per_sample': flips_per_sample.std(),
        'mean_flip_rate': flip_rates.mean(),
        'samples_never_flipped': (flips_per_sample == 0).sum(),
        'samples_highly_unstable': (flips_per_sample > total_epochs * 0.5).sum()
    }
    
    print(f"📊 标签一致性统计:")
    print(f"   - 追踪样本数: {consistency_stats['total_samples_tracked']}")
    print(f"   - 平均翻转次数: {consistency_stats['mean_flips_per_sample']:.2f}")
    print(f"   - 平均翻转率: {consistency_stats['mean_flip_rate']:.2%}")
    print(f"   - 从未翻转样本: {consistency_stats['samples_never_flipped']} ({consistency_stats['samples_never_flipped']/total_samples:.1%})")
    print(f"   - 高度不稳定样本: {consistency_stats['samples_highly_unstable']} ({consistency_stats['samples_highly_unstable']/total_samples:.1%})")
    
    return flips_per_sample, flip_rates, df_pivot, consistency_stats

def create_flip_visualizations(flips_per_sample, df_pivot, output_dir):
    """创建翻转情况的可视化图表"""
    print("📈 创建翻转情况可视化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pseudo-Label Flip Analysis', fontsize=16, fontweight='bold')
    
    # 子图1: 翻转次数分布直方图
    axes[0,0].hist(flips_per_sample, bins=np.arange(0, flips_per_sample.max() + 2) - 0.5, 
                   alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Distribution of Label Flips per Sample')
    axes[0,0].set_xlabel('Number of Flips')
    axes[0,0].set_ylabel('Count of Samples')
    axes[0,0].grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_flips = flips_per_sample.mean()
    axes[0,0].axvline(mean_flips, color='red', linestyle='--', 
                     label=f'Mean: {mean_flips:.2f}')
    axes[0,0].legend()
    
    # 子图2: 每轮epoch的翻转数量
    flips_per_epoch = df_pivot.diff(axis=1).ne(0).sum(axis=0)
    axes[0,1].plot(flips_per_epoch.index, flips_per_epoch.values, 
                   marker='o', color='orange', linewidth=2)
    axes[0,1].set_title('Label Flips per Epoch')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Number of Flips')
    axes[0,1].grid(True, alpha=0.3)
    
    # 子图3: 样本稳定性热图（选择部分样本）
    sample_subset = df_pivot.iloc[:min(20, len(df_pivot))]  # 显示前20个样本
    sns.heatmap(sample_subset, cmap='viridis', cbar_kws={'label': 'Pseudo Label'}, 
                ax=axes[1,0])
    axes[1,0].set_title('Pseudo-Label Evolution (Sample Subset)')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Sample ID')
    
    # 子图4: 翻转累积分布
    flip_counts = flips_per_sample.value_counts().sort_index()
    cumulative_pct = flip_counts.cumsum() / len(flips_per_sample) * 100
    axes[1,1].bar(flip_counts.index, cumulative_pct.values, alpha=0.7, color='lightcoral')
    axes[1,1].set_title('Cumulative Distribution of Label Flips')
    axes[1,1].set_xlabel('Number of Flips')
    axes[1,1].set_ylabel('Cumulative Percentage (%)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # 保存图表
    flip_plot_path = os.path.join(output_dir, '伪标签翻转分析图.png')
    plt.savefig(flip_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ 翻转分析图已保存至: {flip_plot_path}")
    plt.close()
    
    return flip_plot_path

def analyze_dacp_firewall_relationship(df_log, history, output_dir):
    """分析DACP防火墙机制与标签稳定性的关系"""
    if history is None or 'dacp_ema_thresholds' not in history:
        print("⚠️ 缺少DACP阈值数据，跳过防火墙关系分析")
        return None
    
    print("🛡️ 分析DACP防火墙与标签稳定性关系...")
    
    # 准备数据
    df_thresholds = pd.DataFrame(history['dacp_ema_thresholds'])
    warmup_epochs = 30  # 可以从配置获取
    
    # 计算每个epoch的防火墙激活情况
    firewall_activations = (df_thresholds > 1.0).sum(axis=1)
    firewall_epochs = list(range(warmup_epochs + 1, warmup_epochs + 1 + len(firewall_activations)))
    
    # 计算每个epoch的标签翻转情况
    df_pivot = df_log.pivot_table(index='sample_id', columns='epoch', values='pseudo_label')
    flips_per_epoch = df_pivot.diff(axis=1).ne(0).sum(axis=0)
    
    # 对齐数据（只考虑非warmup期间）
    common_epochs = sorted(set(firewall_epochs) & set(flips_per_epoch.index))
    
    if len(common_epochs) < 5:
        print("⚠️ 公共epoch数据不足，跳过相关性分析")
        return None
    
    firewall_aligned = [firewall_activations[e - warmup_epochs - 1] for e in common_epochs]
    flips_aligned = [flips_per_epoch[e] for e in common_epochs]
    
    # 计算相关性
    correlation, p_value = stats.pearsonr(firewall_aligned, flips_aligned)
    
    print(f"🔗 防火墙激活与标签翻转相关性:")
    print(f"   - 皮尔逊相关系数: {correlation:.4f}")
    print(f"   - P值: {p_value:.4f}")
    print(f"   - 显著性: {'显著' if p_value < 0.05 else '不显著'}")
    
    # 创建可视化
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('DACP Firewall vs Label Stability Analysis', fontsize=16, fontweight='bold')
    
    # 子图1: 时间序列对比
    ax1 = axes[0]
    color1 = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Number of Label Flips', color=color1)
    line1 = ax1.plot(common_epochs, flips_aligned, color=color1, marker='o', 
                     label='Label Flips', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Classes with Threshold > 1', color=color2)
    line2 = ax2.plot(common_epochs, firewall_aligned, color=color2, marker='s', 
                     linestyle='--', label='Firewall Activations', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # 添加图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    ax1.set_title(f'Temporal Relationship (Correlation: {correlation:.3f}, p={p_value:.3f})')
    
    # 子图2: 散点图显示相关性
    axes[1].scatter(firewall_aligned, flips_aligned, alpha=0.7, s=60, color='purple')
    axes[1].set_xlabel('Number of Classes with Firewall Activated (τ > 1)')
    axes[1].set_ylabel('Number of Label Flips')
    axes[1].set_title('Correlation Analysis')
    axes[1].grid(True, alpha=0.3)
    
    # 添加拟合线
    if len(firewall_aligned) > 1:
        z = np.polyfit(firewall_aligned, flips_aligned, 1)
        p = np.poly1d(z)
        axes[1].plot(firewall_aligned, p(firewall_aligned), "r--", alpha=0.8, 
                    label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
        axes[1].legend()
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # 保存图表
    firewall_plot_path = os.path.join(output_dir, 'DACP防火墙与标签稳定性关系图.png')
    plt.savefig(firewall_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ 防火墙关系分析图已保存至: {firewall_plot_path}")
    plt.close()
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'plot_path': firewall_plot_path
    }

def analyze_confirmation_patterns(df_log, output_dir):
    """分析确认偏差模式"""
    print("🎯 分析确认偏差模式...")
    
    # 分析不同确定性分数范围的样本稳定性
    df_log['certainty_range'] = pd.cut(df_log['certainty_score'], 
                                      bins=[0, 0.6, 0.8, 0.9, 1.0], 
                                      labels=['Low(0-0.6)', 'Med(0.6-0.8)', 
                                             'High(0.8-0.9)', 'VHigh(0.9-1.0)'])
    
    # 按样本ID和确定性范围分组分析
    pattern_analysis = {}
    
    for certainty_range in df_log['certainty_range'].cat.categories:
        subset = df_log[df_log['certainty_range'] == certainty_range]
        if len(subset) > 0:
            # 计算该确定性范围内的翻转情况
            pivot = subset.pivot_table(index='sample_id', columns='epoch', 
                                     values='pseudo_label', aggfunc='first')
            if len(pivot.columns) > 1:
                flips = pivot.diff(axis=1).ne(0).sum(axis=1)
                pattern_analysis[certainty_range] = {
                    'sample_count': len(pivot),
                    'mean_flips': flips.mean(),
                    'flip_rate': flips.mean() / (len(pivot.columns) - 1) if len(pivot.columns) > 1 else 0
                }
    
    # 可视化确认偏差模式
    if pattern_analysis:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Confirmation Bias Patterns by Certainty Level', 
                     fontsize=16, fontweight='bold')
        
        ranges = list(pattern_analysis.keys())
        mean_flips = [pattern_analysis[r]['mean_flips'] for r in ranges]
        flip_rates = [pattern_analysis[r]['flip_rate'] for r in ranges]
        
        # 平均翻转次数
        axes[0].bar(ranges, mean_flips, color='lightblue', alpha=0.7)
        axes[0].set_title('Mean Flips by Certainty Level')
        axes[0].set_ylabel('Mean Number of Flips')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # 翻转率
        axes[1].bar(ranges, flip_rates, color='lightcoral', alpha=0.7)
        axes[1].set_title('Flip Rate by Certainty Level')
        axes[1].set_ylabel('Flip Rate')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        pattern_plot_path = os.path.join(output_dir, '确认偏差模式分析图.png')
        plt.savefig(pattern_plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ 确认偏差模式图已保存至: {pattern_plot_path}")
        plt.close()
        
        return pattern_analysis, pattern_plot_path
    
    return None, None

def generate_bias_report(consistency_stats, firewall_analysis, pattern_analysis, output_dir):
    """生成确认偏差分析报告"""
    print("📋 生成确认偏差分析报告...")
    
    report_data = {
        'analysis_summary': {
            'analysis_type': 'confirmation_bias_analysis',
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': '通过追踪固定样本伪标签变化分析确认偏差'
        },
        'label_consistency': consistency_stats,
        'firewall_relationship': firewall_analysis if firewall_analysis else {},
        'confirmation_patterns': pattern_analysis if pattern_analysis else {}
    }
    
    # 保存报告
    report_path = os.path.join(output_dir, '确认偏差分析报告.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)
    
    print(f"✅ 确认偏差分析报告已保存至: {report_path}")
    return report_path

def analyze_bias(log_filepath, history_filepath, output_dir):
    """
    主分析函数：执行完整的确认偏差分析
    """
    print("🔍 开始确认偏差分析...")
    
    # 创建输出目录和分析子目录
    analysis_subdir = os.path.join(output_dir, "02_确认偏差深度分析")
    os.makedirs(analysis_subdir, exist_ok=True)
    output_dir = analysis_subdir  # 重定向输出目录
    
    try:
        # 1. 加载数据
        df_log, history = load_analysis_data(log_filepath, history_filepath)
        
        # 2. 分析标签一致性
        flips_per_sample, flip_rates, df_pivot, consistency_stats = analyze_label_consistency(df_log)
        
        # 3. 创建翻转可视化
        flip_plot = create_flip_visualizations(flips_per_sample, df_pivot, output_dir)
        
        # 4. 分析DACP防火墙关系
        firewall_analysis = analyze_dacp_firewall_relationship(df_log, history, output_dir)
        
        # 5. 分析确认偏差模式
        pattern_analysis, pattern_plot = analyze_confirmation_patterns(df_log, output_dir)
        
        # 6. 生成分析报告
        report_path = generate_bias_report(consistency_stats, firewall_analysis, 
                                         pattern_analysis, output_dir)
        
        print("\n🎉 确认偏差分析完成!")
        print(f"📊 生成的文件:")
        print(f"   - 翻转分析图: {flip_plot}")
        if firewall_analysis and 'plot_path' in firewall_analysis:
            print(f"   - 防火墙关系图: {firewall_analysis['plot_path']}")
        if pattern_plot:
            print(f"   - 偏差模式图: {pattern_plot}")
        print(f"   - 分析报告: {report_path}")
        
        return {
            'flip_plot': flip_plot,
            'firewall_analysis': firewall_analysis,
            'pattern_plot': pattern_plot,
            'report': report_path
        }
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='分析模型训练过程中的确认偏差',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    python analyze_confirmation_bias.py --log_path /path/to/confirmation_bias_log.json --history_path /path/to/training_history.json
    python analyze_confirmation_bias.py --log_path /path/to/bias_log.json --history_path /path/to/history.json --output ./bias_analysis
        """
    )
    
    parser.add_argument('--log_path', type=str, required=True,
                       help='确认偏差日志JSON文件路径 (confirmation_bias_log.json)')
    parser.add_argument('--history_path', type=str, default=None,
                       help='训练历史JSON文件路径 (training_history.json, 可选)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出目录路径 (默认为日志文件所在目录)')
    
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output is None:
        output_directory = os.path.dirname(args.log_path)
    else:
        output_directory = args.output
    
    # 执行分析
    try:
        results = analyze_bias(
            log_filepath=args.log_path,
            history_filepath=args.history_path,
            output_dir=output_directory
        )
        print(f"\n✨ 分析完成! 结果已保存至: {output_directory}")
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 