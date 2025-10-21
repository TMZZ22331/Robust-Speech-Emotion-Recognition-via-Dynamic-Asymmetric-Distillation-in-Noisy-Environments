import os
import json
import importlib
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# 配置日志记录
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- 复用消融实验脚本的运行函数 ---
def run_single_experiment(experiment_name, config_overrides):
    """
    运行单次实验，并返回关键指标 WA。
    :param experiment_name: 实验的名称 (用于日志记录)。
    :param config_overrides: 一个字典，包含需要覆盖的配置项。
    :return: 加权准确率 (WA)。
    """
    logger.info(f"--- 🚀 开始敏感性分析实验: {experiment_name} ---")
    
    import config_casia
    importlib.reload(config_casia)
    
    for key, value in config_overrides.items():
        setattr(config_casia, key, value)
        logger.info(f"  🔧 设置参数: {key} = {value}")

    import train_CASIA
    importlib.reload(train_CASIA)

    trainer = train_CASIA.FixedCASIACrossDomainTrainer(fold=3)
    train_results = trainer.train()
    
    results_dir = train_results['results_dir']
    report_path_pattern = os.path.join(results_dir, "reports", "BEST_detailed_results_epoch_*.json")
    
    list_of_files = glob.glob(report_path_pattern)
    if not list_of_files:
        logger.error(f"❌ 在 {results_dir} 中未找到 'BEST' 结果报告。")
        return 0.0
        
    latest_file = max(list_of_files, key=os.path.getctime)
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        report_data = json.load(f)
        
    wa = float(report_data['summary']['noisy']['w_acc'].replace('%', ''))
    
    logger.info(f"--- ✅ 实验结束: {experiment_name} | WA: {wa:.2f}% ---")
    
    return wa

def plot_sensitivity_curve(param_name, param_values, results, save_dir):
    """
    绘制并保存超参数敏感性曲线图。
    :param param_name: 超参数名称。
    :param param_values: 测试的超参数值列表。
    :param results: 对应的性能结果列表 (WA)。
    :param save_dir: 图像保存目录。
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(x=param_values, y=results, marker='o', markersize=8, label=f'WA vs {param_name}')
    
    plt.title(f'Hyperparameter Sensitivity Analysis\n{param_name}', fontsize=16)
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('Weighted Accuracy (WA, %)', fontsize=12)
    plt.xticks(param_values, rotation=45)
    plt.yticks(fontsize=10)
    plt.legend()
    plt.tight_layout()
    
    # 文件名处理，避免特殊字符
    filename = f"sensitivity_{param_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    
    logger.info(f"📈 敏感性分析图已保存至: {save_path}")
    plt.close()


def main():
    """主函数，定义并运行所有超参数敏感性实验"""
    
    # --- 实验定义 ---
    # 定义要分析的超参数及其测试值
    sensitivity_params = {
        'WEIGHT_ECDA': {
            'values': [0.1, 0.3, 0.5, 1.0],
            'name': 'ECDA Loss Weight (λ_ECDA)'
        },
        'DACP_CALIBRATION_STRENGTH_LAMBDA': {
            'values': [0.0, 0.05, 0.1, 0.2],
            'name': 'DACP Calibration Strength (λ)'
        },
        'ECDA_GAMMA_DELTA': {
            'values': [0.0, 0.01, 0.05, 0.1],
            'name': 'ECDA Compactness & Repulsion Weight (γ, δ)'
        }
    }
    
    # --- 结果保存目录 ---
    # 使用一个固定的、带时间戳的目录来保存本次分析的所有图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.path.pardir, os.path.pardir, 'casia_cross_domain_results', 'sensitivity_analysis_plots', timestamp)
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"📊 所有敏感性分析图将保存到: {save_dir}")

    # --- 执行实验 ---
    for param_key, config in sensitivity_params.items():
        param_values = config['values']
        param_name = config['name']
        results = []

        logger.info("\n" + "="*50)
        logger.info(f"🔬 开始分析超参数: {param_name}")
        logger.info("="*50)

        for value in param_values:
            experiment_name = f"{param_name} = {value}"
            
            # 基础配置为完整模型
            overrides = {
                'USE_DACP': True,
                'USE_ECDA': True,
            }
            
            # 特殊处理 gamma 和 delta
            if param_key == 'ECDA_GAMMA_DELTA':
                overrides['ECDA_COMPACTNESS_WEIGHT_GAMMA'] = value
                overrides['ECDA_REPULSION_WEIGHT_DELTA'] = value
            else:
                overrides[param_key] = value
                
            wa_result = run_single_experiment(experiment_name, overrides)
            results.append(wa_result)
        
        # 绘制并保存该参数的敏感性曲线
        plot_sensitivity_curve(param_name, param_values, results, save_dir)
        
    logger.info("\n" + "="*50)
    logger.info("🎉 所有超参数敏感性分析已完成! 🎉")
    logger.info("="*50 + "\n")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main() 