import os
import json
import importlib
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import logging

# 配置日志记录
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_single_experiment(experiment_name, config_overrides, fold):
    """运行单次IEMOCAP实验并返回WA。"""
    logger.info(f"--- 🚀 开始敏感性分析: {experiment_name} (Fold {fold+1}) ---")
    
    import config
    importlib.reload(config)
    
    for key, value in config_overrides.items():
        setattr(config, key, value)
        logger.info(f"  🔧 设置参数: {key} = {value}")

    import train
    importlib.reload(train)

    trainer = train.IEMOCAPCrossDomainTrainer(fold=fold, experiment_name=experiment_name)
    train_results = trainer.train()
    
    results_dir = train_results['results_dir']
    report_path_pattern = os.path.join(results_dir, "reports", "BEST_detailed_results_epoch_*.json")
    
    list_of_files = glob.glob(report_path_pattern)
    if not list_of_files:
        logger.error(f"❌ 在 {results_dir} 中未找到 'BEST' 结果报告。")
        return 0.0
        
    latest_file = max(list_of_files, key=os.path.getctime)
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        wa = float(json.load(f)['summary']['noisy']['w_acc'].replace('%', ''))
    
    logger.info(f"--- ✅ 实验结束: {experiment_name} | WA: {wa:.2f}% ---")
    return wa

def plot_sensitivity_curve(param_display_name, param_values, results, save_dir):
    """绘制并保存超参数敏感性曲线图。"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(x=param_values, y=results, marker='o', markersize=8, label=f'WA vs {param_display_name}')
    
    plt.title(f'IEMOCAP Hyperparameter Sensitivity Analysis\n{param_display_name}', fontsize=16)
    plt.xlabel(param_display_name, fontsize=12)
    plt.ylabel('Weighted Accuracy (WA, %)', fontsize=12)
    plt.xticks(param_values)
    plt.yticks(fontsize=10)
    plt.legend()
    plt.tight_layout()
    
    filename = f"sensitivity_{param_display_name.replace(' ', '_').replace('(', '').replace(')', '').replace('&', 'and')}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    
    logger.info(f"📈 敏感性分析图已保存至: {save_path}")
    plt.close()

def plot_combined_sensitivity_curves(all_results_data, save_dir):
    """绘制合并的敏感性分析曲线图（三条曲线在一张图上）"""
    # 设置论文级别的绘图样式
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))
    
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
    
    # 获取所有参数值的并集作为x轴刻度（假设三个参数使用相同的权重值）
    all_param_values = list(all_results_data.values())[0]['param_values']
    
    # 绘制三条曲线
    for i, (param_key, data) in enumerate(all_results_data.items()):
        param_values = data['param_values']
        results = data['results']
        
        plt.plot(param_values, results, 
                color=colors[i], 
                marker=markers[i], 
                linestyle=linestyles[i],
                markersize=8, 
                linewidth=2.5,
                markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor=colors[i],
                label=legend_labels[i],
                alpha=0.85)
    
    # 设置标题和轴标签
    plt.title('IEMOCAP Hyperparameter Sensitivity Analysis', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Hyperparameter Weight Value', fontsize=14, fontweight='semibold')
    plt.ylabel('Weighted Accuracy (WA, %)', fontsize=14, fontweight='semibold')
    
    # 设置坐标轴 - 使用实际的参数值作为刻度
    plt.xlim(-0.05, 1.05)  # 给横轴留一点边距
    plt.xticks(all_param_values, fontsize=12)  # 使用实际参数值作为刻度
    plt.yticks(fontsize=12)
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 设置图例（放在右上角）
    legend = plt.legend(loc='upper right', 
                       frameon=True, 
                       fancybox=True, 
                       shadow=True,
                       fontsize=12,
                       framealpha=0.95,
                       edgecolor='black',
                       facecolor='white',
                       borderpad=1.0,
                       columnspacing=1.0,
                       handlelength=2.0,
                       handletextpad=0.8,
                       labelspacing=0.8)
    
    # 美化图例边框
    legend.get_frame().set_linewidth(1.2)
    
    # 设置布局
    plt.tight_layout()
    
    # 保存图片
    filename = "combined_sensitivity_analysis_IEMOCAP.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    logger.info(f"📈 合并敏感性分析图已保存至: {save_path}")
    plt.close()

    # 额外保存PDF格式（适合论文使用）
    pdf_path = os.path.join(save_dir, "combined_sensitivity_analysis_IEMOCAP.pdf")
    plt.figure(figsize=(12, 8))
    
    # 重新绘制（为了PDF格式）
    for i, (param_key, data) in enumerate(all_results_data.items()):
        param_values = data['param_values']
        results = data['results']
        
        plt.plot(param_values, results, 
                color=colors[i], 
                marker=markers[i], 
                linestyle=linestyles[i],
                markersize=8, 
                linewidth=2.5,
                markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor=colors[i],
                label=legend_labels[i],
                alpha=0.85)
    
    plt.title('IEMOCAP Hyperparameter Sensitivity Analysis', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Hyperparameter Weight Value', fontsize=14, fontweight='semibold')
    plt.ylabel('Weighted Accuracy (WA, %)', fontsize=14, fontweight='semibold')
    plt.xlim(-0.05, 1.05)
    plt.xticks(all_param_values, fontsize=12)  # 使用实际参数值作为刻度
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    legend = plt.legend(loc='upper right', 
                       frameon=True, 
                       fancybox=True, 
                       shadow=True,
                       fontsize=12,
                       framealpha=0.95,
                       edgecolor='black',
                       facecolor='white',
                       borderpad=1.0,
                       columnspacing=1.0,
                       handlelength=2.0,
                       handletextpad=0.8,
                       labelspacing=0.8)
    legend.get_frame().set_linewidth(1.2)
    
    plt.tight_layout()
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf', facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"📈 PDF格式敏感性分析图已保存至: {pdf_path}")

def main():
    """主函数，定义并运行所有IEMOCAP超参数敏感性实验"""
    TARGET_FOLD = 3 # IEMOCAP 使用第4折 (fold=3)

    sensitivity_params = {
        'WEIGHT_ECDA': {'values': [0.0,0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'name': 'ECDA Loss Weight (λ_ECDA)'},
        'DACP_CALIBRATION_STRENGTH_LAMBDA': {'values': [0.0,0.01, 0.05, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], 'name': 'DACP Calibration Strength (λ)'},
        'ECDA_GAMMA_DELTA': {'values': [0.0, 0.01, 0.05, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], 'name': 'ECDA Compactness & Repulsion Weight (γ, δ)'}
    }
    
    all_results = {}
    
    for param_key, config in sensitivity_params.items():
        param_values = config['values']
        param_display_name = config['name']
        results = []
        
        # 为每个超参数组创建一个独立的父目录名称（不包含路径分隔符）
        safe_param_name = re.sub(r'[\s/\\&().,]', '_', param_display_name)
        
        logger.info("\n" + "="*50)
        logger.info(f"🔬 开始分析IEMOCAP超参数: {param_display_name}")
        logger.info("="*50)

        for value in param_values:
            # 每个实验点都有自己独立的名称（移除路径分隔符）
            experiment_name = f"Sensitivity_Analysis_{safe_param_name}__value_{value}"
            
            overrides = {'USE_DACP': True, 'USE_ECDA': True}
            
            if param_key == 'ECDA_GAMMA_DELTA':
                overrides['ECDA_COMPACTNESS_WEIGHT_GAMMA'] = value
                overrides['ECDA_REPULSION_WEIGHT_DELTA'] = value
            else:
                overrides[param_key] = value
                
            wa_result = run_single_experiment(experiment_name, overrides, fold=TARGET_FOLD)
            results.append(wa_result)
        
        # 保存当前参数的结果
        all_results[param_key] = {
            'param_display_name': param_display_name,
            'param_values': param_values,
            'results': results,
            'safe_param_name': safe_param_name
        }
        
    # 在所有实验完成后，生成敏感性分析图
    logger.info("\n" + "="*50)
    logger.info("📊 生成敏感性分析图表...")
    logger.info("="*50)
    
    # 创建统一的plots保存目录
    base_results_dir = "iemocap_cross_domain_results"
    sensitivity_plots_dir = os.path.join(base_results_dir, "Sensitivity_Analysis_IEMOCAP", "plots")
    os.makedirs(sensitivity_plots_dir, exist_ok=True)
    
    plot_combined_sensitivity_curves(all_results, sensitivity_plots_dir)
        
    logger.info("\n" + "="*50)
    logger.info("🎉 所有IEMOCAP超参数敏感性分析已完成! 🎉")
    logger.info(f"📈 敏感性分析图表保存在: {sensitivity_plots_dir}")
    logger.info("="*50 + "\n")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main() 