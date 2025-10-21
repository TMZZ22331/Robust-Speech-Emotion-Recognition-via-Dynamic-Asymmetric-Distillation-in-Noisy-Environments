#!/usr/bin/env python3
"""
教师-学生网络差异性分析脚本
分析教师网络和学生网络在预测上的差异性（Disagreement）
揭示知识蒸馏的动态过程和模型一致性演变
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix

def load_training_history(history_filepath):
    """加载训练历史数据"""
    if not os.path.exists(history_filepath):
        raise FileNotFoundError(f"训练历史文件不存在: {history_filepath}")
    
    with open(history_filepath, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    print(f"✅ 成功加载训练历史数据: {history_filepath}")
    return history

def extract_disagreement_data(history, validation_interval=5, warmup_epochs=30):
    """提取并处理差异性数据"""
    print("📊 提取教师-学生网络差异性数据...")
    
    # 检查必要的数据
    required_keys = [
        'disagreement_rate_noisy', 'disagreement_rate_clean',
        'dacp_ema_thresholds', 'dacp_class_quality'
    ]
    
    available_keys = [key for key in required_keys if key in history and history[key]]
    missing_keys = [key for key in required_keys if key not in available_keys]
    
    if 'disagreement_rate_noisy' not in available_keys:
        raise ValueError("训练历史中缺少关键数据 'disagreement_rate_noisy'")
    
    if missing_keys:
        print(f"⚠️ 部分数据缺失，将跳过相关分析: {missing_keys}")
    
    # 提取差异率数据
    noisy_disagreement = history['disagreement_rate_noisy']
    clean_disagreement = history.get('disagreement_rate_clean', [])
    
    # 计算对应的epoch
    num_validations_noisy = len(noisy_disagreement)
    num_validations_clean = len(clean_disagreement)
    
    # 验证通常从warmup结束后开始，每隔validation_interval进行一次
    epochs_noisy = [warmup_epochs + (i + 1) * validation_interval for i in range(num_validations_noisy)]
    epochs_clean = [warmup_epochs + (i + 1) * validation_interval for i in range(num_validations_clean)]
    
    # 创建数据框
    df_noisy = pd.DataFrame({
        'epoch': epochs_noisy,
        'disagreement_rate': noisy_disagreement,
        'domain': 'noisy'
    })
    
    df_clean = pd.DataFrame({
        'epoch': epochs_clean,
        'disagreement_rate': clean_disagreement,
        'domain': 'clean'
    }) if clean_disagreement else pd.DataFrame()
    
    print(f"📈 差异性数据统计:")
    print(f"   - 噪声域验证点: {num_validations_noisy} 个")
    print(f"   - 干净域验证点: {num_validations_clean} 个")
    print(f"   - Epoch范围: {min(epochs_noisy)} - {max(epochs_noisy)}")
    
    return df_noisy, df_clean, available_keys

def analyze_disagreement_trends(df_noisy, df_clean=None):
    """分析差异性趋势"""
    print("📉 分析差异性趋势...")
    
    # 分析噪声域趋势
    noisy_stats = {
        'mean_disagreement': df_noisy['disagreement_rate'].mean(),
        'std_disagreement': df_noisy['disagreement_rate'].std(),
        'max_disagreement': df_noisy['disagreement_rate'].max(),
        'min_disagreement': df_noisy['disagreement_rate'].min(),
        'final_disagreement': df_noisy['disagreement_rate'].iloc[-1],
        'initial_disagreement': df_noisy['disagreement_rate'].iloc[0]
    }
    
    # 计算趋势斜率
    if len(df_noisy) > 1:
        x = np.arange(len(df_noisy))
        y = df_noisy['disagreement_rate'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        noisy_stats.update({
            'trend_slope': slope,
            'trend_r_squared': r_value**2,
            'trend_p_value': p_value
        })
    
    # 分析干净域趋势（如果有数据）
    clean_stats = {}
    if df_clean is not None and len(df_clean) > 0:
        clean_stats = {
            'mean_disagreement': df_clean['disagreement_rate'].mean(),
            'std_disagreement': df_clean['disagreement_rate'].std(),
            'max_disagreement': df_clean['disagreement_rate'].max(),
            'min_disagreement': df_clean['disagreement_rate'].min(),
            'final_disagreement': df_clean['disagreement_rate'].iloc[-1],
            'initial_disagreement': df_clean['disagreement_rate'].iloc[0]
        }
        
        if len(df_clean) > 1:
            x = np.arange(len(df_clean))
            y = df_clean['disagreement_rate'].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            clean_stats.update({
                'trend_slope': slope,
                'trend_r_squared': r_value**2,
                'trend_p_value': p_value
            })
    
    print(f"🎯 噪声域差异性统计:")
    print(f"   - 平均差异率: {noisy_stats['mean_disagreement']:.3f}")
    print(f"   - 最大差异率: {noisy_stats['max_disagreement']:.3f}")
    print(f"   - 最小差异率: {noisy_stats['min_disagreement']:.3f}")
    
    if 'trend_slope' in noisy_stats:
        trend_direction = "上升" if noisy_stats['trend_slope'] > 0 else "下降"
        print(f"   - 趋势: {trend_direction} (斜率: {noisy_stats['trend_slope']:.6f})")
        print(f"   - R²: {noisy_stats['trend_r_squared']:.3f}")
    
    if clean_stats:
        print(f"🎯 干净域差异性统计:")
        print(f"   - 平均差异率: {clean_stats['mean_disagreement']:.3f}")
        if 'trend_slope' in clean_stats:
            trend_direction = "上升" if clean_stats['trend_slope'] > 0 else "下降"
            print(f"   - 趋势: {trend_direction} (斜率: {clean_stats['trend_slope']:.6f})")
    
    return noisy_stats, clean_stats

def create_disagreement_plots(df_noisy, df_clean, output_dir):
    """创建差异性分析图表"""
    print("🎨 创建差异性分析图表...")
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # 合并数据用于绘图
    df_combined = pd.concat([df_noisy, df_clean]) if len(df_clean) > 0 else df_noisy
    
    # 创建主图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Teacher-Student Network Disagreement Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 子图1: 时间序列图
    if len(df_clean) > 0:
        sns.lineplot(data=df_combined, x='epoch', y='disagreement_rate', 
                    hue='domain', marker='o', linewidth=2.5, ax=axes[0,0])
        axes[0,0].set_title('Disagreement Rate Evolution (Both Domains)')
    else:
        axes[0,0].plot(df_noisy['epoch'], df_noisy['disagreement_rate'], 
                      marker='o', color='orange', linewidth=2.5, label='Noisy Domain')
        axes[0,0].set_title('Disagreement Rate Evolution (Noisy Domain)')
        axes[0,0].legend()
    
    axes[0,0].set_xlabel('Training Epoch')
    axes[0,0].set_ylabel('Disagreement Rate')
    axes[0,0].grid(True, alpha=0.3)
    
    # 添加趋势线
    if len(df_noisy) > 1:
        z = np.polyfit(df_noisy['epoch'], df_noisy['disagreement_rate'], 1)
        p = np.poly1d(z)
        axes[0,0].plot(df_noisy['epoch'], p(df_noisy['epoch']), 
                      "r--", alpha=0.8, label=f'Noisy Trend: y={z[0]:.6f}x+{z[1]:.3f}')
        axes[0,0].legend()
    
    # 子图2: 分布直方图
    axes[0,1].hist(df_noisy['disagreement_rate'], bins=15, alpha=0.7, 
                  color='orange', edgecolor='black', label='Noisy Domain')
    if len(df_clean) > 0:
        axes[0,1].hist(df_clean['disagreement_rate'], bins=15, alpha=0.7, 
                      color='blue', edgecolor='black', label='Clean Domain')
    
    axes[0,1].set_title('Disagreement Rate Distribution')
    axes[0,1].set_xlabel('Disagreement Rate')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 子图3: 箱线图对比
    if len(df_clean) > 0:
        sns.boxplot(data=df_combined, x='domain', y='disagreement_rate', ax=axes[1,0])
        axes[1,0].set_title('Disagreement Rate Comparison')
    else:
        axes[1,0].boxplot(df_noisy['disagreement_rate'])
        axes[1,0].set_title('Disagreement Rate Distribution (Noisy)')
        axes[1,0].set_xticklabels(['Noisy Domain'])
    
    axes[1,0].set_ylabel('Disagreement Rate')
    axes[1,0].grid(True, alpha=0.3)
    
    # 子图4: 移动平均图
    window_size = min(5, len(df_noisy) // 2)
    if window_size > 1:
        df_noisy_ma = df_noisy.copy()
        df_noisy_ma['disagreement_ma'] = df_noisy_ma['disagreement_rate'].rolling(window=window_size).mean()
        
        axes[1,1].plot(df_noisy['epoch'], df_noisy['disagreement_rate'], 
                      'o-', alpha=0.5, color='orange', label='Raw Data')
        axes[1,1].plot(df_noisy_ma['epoch'], df_noisy_ma['disagreement_ma'], 
                      's-', color='red', linewidth=2, label=f'Moving Average (window={window_size})')
        
        if len(df_clean) > 0 and len(df_clean) > window_size:
            df_clean_ma = df_clean.copy()
            df_clean_ma['disagreement_ma'] = df_clean_ma['disagreement_rate'].rolling(window=window_size).mean()
            axes[1,1].plot(df_clean['epoch'], df_clean['disagreement_rate'], 
                          'o-', alpha=0.5, color='blue', label='Clean Raw')
            axes[1,1].plot(df_clean_ma['epoch'], df_clean_ma['disagreement_ma'], 
                          's-', color='navy', linewidth=2, label=f'Clean MA (window={window_size})')
    else:
        axes[1,1].plot(df_noisy['epoch'], df_noisy['disagreement_rate'], 
                      'o-', color='orange', label='Noisy Domain')
        if len(df_clean) > 0:
            axes[1,1].plot(df_clean['epoch'], df_clean['disagreement_rate'], 
                          'o-', color='blue', label='Clean Domain')
    
    axes[1,1].set_title('Smoothed Disagreement Trends')
    axes[1,1].set_xlabel('Training Epoch')
    axes[1,1].set_ylabel('Disagreement Rate')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # 保存图表
    main_plot_path = os.path.join(output_dir, '教师学生网络差异分析图.png')
    plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ 主要差异性分析图已保存至: {main_plot_path}")
    plt.close()
    
    return main_plot_path

def analyze_disagreement_vs_mechanisms(df_noisy, history, output_dir):
    """分析差异性与DACP/ECDA机制的关系"""
    if 'dacp_ema_thresholds' not in history:
        print("⚠️ 缺少DACP数据，跳过机制关联分析")
        return None
    
    print("🔗 分析差异性与训练机制的关系...")
    
    # 提取DACP数据
    dacp_thresholds = np.array(history['dacp_ema_thresholds'])
    dacp_quality = np.array(history['dacp_class_quality']) if 'dacp_class_quality' in history else None
    
    # 计算各种DACP指标
    firewall_activations = (dacp_thresholds > 1.0).sum(axis=1)  # 每个epoch防火墙激活的类别数
    mean_thresholds = dacp_thresholds.mean(axis=1)  # 每个epoch的平均阈值
    threshold_std = dacp_thresholds.std(axis=1)  # 每个epoch的阈值标准差
    
    # 对齐数据 - 验证通常每5个epoch进行一次
    validation_interval = 5
    warmup_epochs = 30
    
    # 找到对应的DACP数据点
    disagreement_epochs = df_noisy['epoch'].values
    dacp_indices = [(epoch - warmup_epochs - 1) // validation_interval 
                   for epoch in disagreement_epochs 
                   if (epoch - warmup_epochs - 1) >= 0 and (epoch - warmup_epochs - 1) // validation_interval < len(firewall_activations)]
    
    # 过滤有效的数据点
    valid_indices = [i for i in dacp_indices if 0 <= i < len(firewall_activations)]
    valid_disagreement = df_noisy.iloc[:len(valid_indices)]['disagreement_rate'].values
    
    if len(valid_indices) < 3:
        print("⚠️ 有效数据点不足，跳过相关性分析")
        return None
    
    aligned_firewall = firewall_activations[valid_indices]
    aligned_mean_thresholds = mean_thresholds[valid_indices]
    aligned_threshold_std = threshold_std[valid_indices]
    
    # 计算相关性
    correlations = {}
    correlations['firewall_vs_disagreement'] = stats.pearsonr(aligned_firewall, valid_disagreement)
    correlations['mean_threshold_vs_disagreement'] = stats.pearsonr(aligned_mean_thresholds, valid_disagreement)
    correlations['threshold_std_vs_disagreement'] = stats.pearsonr(aligned_threshold_std, valid_disagreement)
    
    # 打印相关性结果
    print("🔍 差异性与训练机制相关性分析:")
    for key, (corr, p_val) in correlations.items():
        significance = "显著" if p_val < 0.05 else "不显著"
        print(f"   - {key}: r={corr:.4f}, p={p_val:.4f} ({significance})")
    
    # 创建关联分析图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Disagreement vs Training Mechanisms Correlation', 
                 fontsize=16, fontweight='bold')
    
    # 防火墙激活 vs 差异性
    axes[0,0].scatter(aligned_firewall, valid_disagreement, alpha=0.7, s=60, color='red')
    axes[0,0].set_xlabel('Firewall Activations (Classes with τ > 1)')
    axes[0,0].set_ylabel('Disagreement Rate')
    axes[0,0].set_title(f'Firewall vs Disagreement (r={correlations["firewall_vs_disagreement"][0]:.3f})')
    axes[0,0].grid(True, alpha=0.3)
    
    # 平均阈值 vs 差异性
    axes[0,1].scatter(aligned_mean_thresholds, valid_disagreement, alpha=0.7, s=60, color='blue')
    axes[0,1].set_xlabel('Mean DACP Threshold')
    axes[0,1].set_ylabel('Disagreement Rate')
    axes[0,1].set_title(f'Mean Threshold vs Disagreement (r={correlations["mean_threshold_vs_disagreement"][0]:.3f})')
    axes[0,1].grid(True, alpha=0.3)
    
    # 阈值标准差 vs 差异性
    axes[1,0].scatter(aligned_threshold_std, valid_disagreement, alpha=0.7, s=60, color='green')
    axes[1,0].set_xlabel('Threshold Std Dev')
    axes[1,0].set_ylabel('Disagreement Rate')
    axes[1,0].set_title(f'Threshold Diversity vs Disagreement (r={correlations["threshold_std_vs_disagreement"][0]:.3f})')
    axes[1,0].grid(True, alpha=0.3)
    
    # 时间序列对比
    valid_epochs = df_noisy.iloc[:len(valid_indices)]['epoch'].values
    ax1 = axes[1,1]
    color1 = 'tab:orange'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Disagreement Rate', color=color1)
    line1 = ax1.plot(valid_epochs, valid_disagreement, 'o-', color=color1, label='Disagreement')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Firewall Activations', color=color2)
    line2 = ax2.plot(valid_epochs, aligned_firewall, 's--', color=color2, label='Firewall')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # 添加图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.set_title('Temporal Relationship')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # 保存图表
    correlation_plot_path = os.path.join(output_dir, '差异性与训练机制关联图.png')
    plt.savefig(correlation_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ 机制关联分析图已保存至: {correlation_plot_path}")
    plt.close()
    
    return {
        'correlations': correlations,
        'plot_path': correlation_plot_path
    }

def generate_disagreement_report(noisy_stats, clean_stats, correlation_analysis, output_dir):
    """生成差异性分析报告"""
    print("📋 生成差异性分析报告...")
    
    report_data = {
        'analysis_summary': {
            'analysis_type': 'teacher_student_disagreement_analysis',
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': '分析教师网络和学生网络预测差异性的动态演变'
        },
        'noisy_domain_statistics': noisy_stats,
        'clean_domain_statistics': clean_stats,
        'mechanism_correlations': {}
    }
    
    if correlation_analysis:
        # 转换相关性数据为可序列化格式
        correlations_serializable = {}
        for key, (corr, p_val) in correlation_analysis['correlations'].items():
            correlations_serializable[key] = {
                'correlation': float(corr) if not np.isnan(corr) else None,
                'p_value': float(p_val) if not np.isnan(p_val) else None,
                'significant': bool(p_val < 0.05) if not np.isnan(p_val) else False
            }
        report_data['mechanism_correlations'] = correlations_serializable
    
    # 保存报告
    report_path = os.path.join(output_dir, '教师学生网络差异分析报告.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)
    
    print(f"✅ 差异性分析报告已保存至: {report_path}")
    return report_path

def analyze_disagreement(history_filepath, output_dir, validation_interval=5, warmup_epochs=30):
    """
    主分析函数：执行完整的教师-学生网络差异性分析
    """
    print("🤝 开始教师-学生网络差异性分析...")
    
    # 创建输出目录和分析子目录
    analysis_subdir = os.path.join(output_dir, "03_教师学生网络差异分析")
    os.makedirs(analysis_subdir, exist_ok=True)
    output_dir = analysis_subdir  # 重定向输出目录
    
    try:
        # 1. 加载数据
        history = load_training_history(history_filepath)
        
        # 2. 提取差异性数据
        df_noisy, df_clean, available_keys = extract_disagreement_data(
            history, validation_interval, warmup_epochs
        )
        
        # 3. 分析趋势
        noisy_stats, clean_stats = analyze_disagreement_trends(df_noisy, df_clean)
        
        # 4. 创建可视化
        main_plot = create_disagreement_plots(df_noisy, df_clean, output_dir)
        
        # 5. 分析与训练机制的关系
        correlation_analysis = analyze_disagreement_vs_mechanisms(df_noisy, history, output_dir)
        
        # 6. 生成分析报告
        report_path = generate_disagreement_report(
            noisy_stats, clean_stats, correlation_analysis, output_dir
        )
        
        print("\n🎉 教师-学生网络差异性分析完成!")
        print(f"📊 生成的文件:")
        print(f"   - 主要分析图: {main_plot}")
        if correlation_analysis:
            print(f"   - 机制关联图: {correlation_analysis['plot_path']}")
        print(f"   - 分析报告: {report_path}")
        
        return {
            'main_plot': main_plot,
            'correlation_analysis': correlation_analysis,
            'report': report_path
        }
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='分析教师-学生网络的预测差异性',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    python analyze_disagreement.py --path /path/to/training_history.json
    python analyze_disagreement.py --path /path/to/training_history.json --output ./disagreement_analysis
    python analyze_disagreement.py --path /path/to/training_history.json --validation_interval 3 --warmup 20
        """
    )
    
    parser.add_argument('--path', type=str, required=True,
                       help='训练历史JSON文件路径 (training_history.json)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出目录路径 (默认为历史文件所在目录)')
    parser.add_argument('--validation_interval', type=int, default=5,
                       help='验证间隔轮次 (默认: 5)')
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
        results = analyze_disagreement(
            history_filepath=args.path,
            output_dir=output_directory,
            validation_interval=args.validation_interval,
            warmup_epochs=args.warmup
        )
        print(f"\n✨ 分析完成! 结果已保存至: {output_directory}")
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 