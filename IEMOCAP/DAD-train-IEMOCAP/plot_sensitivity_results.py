#!/usr/bin/env python3
"""
IEMOCAP敏感性分析结果绘图脚本
读取已完成的敏感性分析实验的JSON结果文件，重新绘制美观的图表
解决图例遮挡和标签重叠问题
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_sensitivity_results():
    """读取敏感性分析实验的JSON结果文件"""
    base_results_dir = r"C:\Users\admin\Desktop\111\good_-emo\IEMOCAP\DAD-train-IEMOCAP\iemocap_cross_domain_results"
    
    # 定义参数映射
    param_mappings = {
        'ECDA_Loss_Weight__λ_ECDA_': {
            'display_name': 'ECDA Loss Weight (λ_ECDA)',
            'values': [0.0,0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        },
        'DACP_Calibration_Strength__λ_': {
            'display_name': 'DACP Calibration Strength (λ)',
            'values': [0.0,0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        },
        'ECDA_Compactness___Repulsion_Weight__γ__δ_': {
            'display_name': 'ECDA Compactness & Repulsion Weight (γ, δ)',
            'values': [0.0,0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }
    }
    
    all_results = {}
    
    for param_key, param_info in param_mappings.items():
        param_values = param_info['values']
        display_name = param_info['display_name']
        results = []
        
        logger.info(f"读取参数: {display_name}")
        
        for value in param_values:
            # 构建实验目录路径
            experiment_name = f"Sensitivity_Analysis_{param_key}__value_{value}"
            experiment_dir = os.path.join(base_results_dir, experiment_name, "fold_4")
            reports_dir = os.path.join(experiment_dir, "reports")
            
            if not os.path.exists(reports_dir):
                logger.warning(f"未找到报告目录: {reports_dir}")
                results.append(0.0)  # 使用默认值
                continue
            
            # 查找BEST结果文件
            report_pattern = os.path.join(reports_dir, "BEST_detailed_results_epoch_*.json")
            report_files = glob.glob(report_pattern)
            
            if not report_files:
                logger.warning(f"未找到BEST结果文件: {reports_dir}")
                results.append(0.0)  # 使用默认值
                continue
            
            # 使用最新的文件
            latest_file = max(report_files, key=os.path.getctime)
            
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    wa_str = data['summary']['noisy']['w_acc']
                    wa_value = float(wa_str.replace('%', ''))
                    results.append(wa_value)
                    logger.info(f"  值 {value}: WA = {wa_value:.2f}%")
            except Exception as e:
                logger.error(f"读取文件失败 {latest_file}: {e}")
                results.append(0.0)
        
        all_results[param_key] = {
            'display_name': display_name,
            'values': param_values,
            'results': results
        }
    
    return all_results

def plot_improved_sensitivity_curves(all_results, save_dir):
    """绘制改进的敏感性分析曲线图"""
    # 设置论文级别的绘图样式
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 10))  # 增大图片尺寸
    
    # 定义三种深色且区分明显的颜色
    colors = ['#1f77b4', '#d62728', '#2ca02c']  # 深蓝, 深红, 深绿
    markers = ['o', 's', '^']  # 圆点, 方形, 三角形
    linestyles = ['-', '--', '-.']  # 实线, 虚线, 点划线
    
    # 定义简化的图例标签
    legend_labels = [
        'ECDA Loss Weight (λ_ECDA)',
        'DACP Calibration Strength (λ)', 
        'ECDA Compactness & Repulsion Weight (γ, δ)'
    ]
    
    # 获取参数值
    param_values = list(all_results.values())[0]['values']
    
    # 绘制三条曲线
    for i, (param_key, data) in enumerate(all_results.items()):
        values = data['values']
        results = data['results']
        
        ax.plot(values, results, 
                color=colors[i], 
                marker=markers[i], 
                linestyle=linestyles[i],
                markersize=10, 
                linewidth=3.0,
                markerfacecolor='white',
                markeredgewidth=2.5,
                markeredgecolor=colors[i],
                label=legend_labels[i],
                alpha=0.9)
    
    # 设置标题和轴标签
    ax.set_title('IEMOCAP 10db Hyperparameter Sensitivity Analysis', 
                 fontsize=20, fontweight='bold', pad=25)
    ax.set_xlabel('Hyperparameter Weight Value', fontsize=16, fontweight='semibold')
    ax.set_ylabel('Weighted Accuracy (WA, %)', fontsize=16, fontweight='semibold')
    
    # 设置坐标轴
    ax.set_xlim(-0.02, 1.02)  # 减少边距避免标签被切割
    
    # 解决x轴标签重叠问题 - 旋转标签并调整字体
    ax.set_xticks(param_values)
    ax.set_xticklabels([f'{v:.2f}' for v in param_values], 
                       fontsize=11, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=13)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 将图例放在左上角，避免遮挡曲线
    legend = ax.legend(loc='lower right', #upper right,down right
                       frameon=True, 
                       fancybox=True, 
                       shadow=True,
                       fontsize=13,
                       framealpha=0.95,
                       edgecolor='black',
                       facecolor='white',
                       borderpad=1.2,
                       columnspacing=1.0,
                       handlelength=2.5,
                       handletextpad=0.8,
                       labelspacing=0.9)
    
    # 美化图例边框
    legend.get_frame().set_linewidth(1.2)
    
    # 调整布局，为旋转的标签留出空间
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # 为旋转的x轴标签留出空间
    
    # 保存PNG格式
    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, "improved_sensitivity_analysis_IEMOCAP.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    logger.info(f"📈 改进版敏感性分析图已保存至: {png_path}")
    
    # 保存PDF格式
    pdf_path = os.path.join(save_dir, "improved_sensitivity_analysis_IEMOCAP.pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf', facecolor='white', edgecolor='none')
    logger.info(f"📈 PDF格式改进版敏感性分析图已保存至: {pdf_path}")
    
    plt.close()

def main():
    """主函数"""
    logger.info("🚀 开始重新绘制IEMOCAP敏感性分析图表...")
    
    # 读取结果数据
    all_results = read_sensitivity_results()
    
    if not all_results:
        logger.error("❌ 未能读取到任何结果数据！")
        return
    
    # 打印读取到的数据概要
    logger.info("\n📊 数据读取概要:")
    for param_key, data in all_results.items():
        logger.info(f"  {data['display_name']}: {len(data['results'])} 个数据点")
        logger.info(f"    WA范围: {min(data['results']):.2f}% - {max(data['results']):.2f}%")
    
    # 创建保存目录
    save_dir = os.path.join("iemocap_cross_domain_results", "Sensitivity_Analysis_IEMOCAP", "plots")
    
    # 绘制改进的图表
    plot_improved_sensitivity_curves(all_results, save_dir)
    
    logger.info("🎉 IEMOCAP敏感性分析图表重新绘制完成！")

if __name__ == "__main__":
    main() 