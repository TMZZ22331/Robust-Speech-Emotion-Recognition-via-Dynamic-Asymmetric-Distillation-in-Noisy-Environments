import os
import json
import importlib
import glob
import pandas as pd
import re
import logging

# 配置日志记录
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_single_experiment(experiment_name, config_overrides, fold):
    """
    运行单次IEMOCAP实验。
    :param experiment_name: 实验名称，用于创建独立的结果目录。
    :param config_overrides: 需要覆盖的配置项。
    :param fold: 交叉验证的折数。
    :return: 一个包含 'WA' 和 'W-F1' 的字典。
    """
    logger.info(f"--- 🚀 开始IEMOCAP实验: {experiment_name} (Fold {fold+1}) ---")
    
    # --- 动态加载和修改配置 ---
    import config
    importlib.reload(config)
    
    for key, value in config_overrides.items():
        setattr(config, key, value)
        logger.info(f"  🔧 设置参数: {key} = {value}")
    
    # --- ⚠️ 关键修复2：更新依赖路径 ---
    if 'NOISY_DATA_DIR' in config_overrides:
        import os
        config.NOISY_FEAT_PATH = os.path.join(config.NOISY_DATA_DIR, "train")
        logger.info(f"  📁 噪声特征路径已更新: {config.NOISY_FEAT_PATH}")
    
    # --- ⚠️ 关键修复1：设置随机种子和环境 ---
    config.setup_environment()
    logger.info(f"  🎲 随机种子已设置: {config.RANDOM_SEED}")

    # --- 运行训练 ---
    import train
    importlib.reload(train)
    
    trainer = train.IEMOCAPCrossDomainTrainer(fold=fold, experiment_name=experiment_name)
    train_results = trainer.train()
    
    # --- 提取结果 ---
    results_dir = train_results['results_dir']
    report_path_pattern = os.path.join(results_dir, "reports", "BEST_detailed_results_epoch_*.json")
    
    list_of_files = glob.glob(report_path_pattern)
    if not list_of_files:
        logger.error(f"❌ 在 {results_dir} 中未找到 'BEST' 结果报告。")
        return {'WA (%)': 'Error', 'W-F1 (%)': 'Error'}
        
    latest_file = max(list_of_files, key=os.path.getctime)
    
    logger.info(f"  📊 读取报告: {os.path.basename(latest_file)}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        report_data = json.load(f)
        
    noisy_summary = report_data['summary']['noisy']
    wa = float(noisy_summary['w_acc'].replace('%', ''))
    w_f1 = float(noisy_summary['w_f1'].replace('%', ''))
    
    logger.info(f"--- ✅ 实验结束: {experiment_name} (Fold {fold+1}) | WA: {wa:.2f}%, W-F1: {w_f1:.2f}% ---")
    
    return {'WA (%)': wa, 'W-F1 (%)': w_f1}

def main():
    """主函数，定义并运行所有IEMOCAP消融实验"""
    
    # --- 实验定义 ---
    # IEMOCAP 使用第2折 (fold=1)
    TARGET_FOLD = 1 

    core_ablations = [
        # ... (此处省略了完整模型，因为您已做过)
        {
            'name': 'emotion2vec+ssl babble 0db',
            'settings': {'USE_DACP': False, 'USE_ECDA': False, 'FIXED_CONFIDENCE_THRESHOLD': 0.75, 
                         'NOISY_DATA_DIR': r"C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP_noisy\root1-babble-0db"}
        },
        {
            'name': 'emotion2vec+ssl babble 10db',
            'settings': {'USE_DACP': False, 'USE_ECDA': False, 'FIXED_CONFIDENCE_THRESHOLD': 0.75, 
                         'NOISY_DATA_DIR': r"C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP_noisy\root1-babble-10db"}
        },
        {
            'name': 'emotion2vec+ssl babble 15db',
            'settings': {'USE_DACP': False, 'USE_ECDA': False, 'FIXED_CONFIDENCE_THRESHOLD': 0.75, 
                         'NOISY_DATA_DIR': r"C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP_noisy\root1-babble-15db"}
        },
        {
            'name': 'emotion2vec+ssl babble 20db',
            'settings': {'USE_DACP': False, 'USE_ECDA': False, 'FIXED_CONFIDENCE_THRESHOLD': 0.75, 
                         'NOISY_DATA_DIR': r"C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP_noisy\root1-babble-20db"}
        },
        {
            'name': 'emotion2vec+ssl factory 0db',
            'settings': {'USE_DACP': False, 'USE_ECDA': False, 'FIXED_CONFIDENCE_THRESHOLD': 0.75, 
                         'NOISY_DATA_DIR': r"C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP_noisy\root1-factory-0db"}
        },
        {
            'name': 'emotion2vec+ssl factory 10db',
            'settings': {'USE_DACP': False, 'USE_ECDA': False, 'FIXED_CONFIDENCE_THRESHOLD': 0.75, 
                         'NOISY_DATA_DIR': r"C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP_noisy\root1-factory-10db"}
        },
        {
            'name': 'emotion2vec+ssl factory 15db',
            'settings': {'USE_DACP': False, 'USE_ECDA': False, 'FIXED_CONFIDENCE_THRESHOLD': 0.75, 
                         'NOISY_DATA_DIR': r"C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP_noisy\root1-factory-15db"}
        },
        {
            'name': 'emotion2vec+ssl factory 20db',
            'settings': {'USE_DACP': False, 'USE_ECDA': False, 'FIXED_CONFIDENCE_THRESHOLD': 0.75, 
                         'NOISY_DATA_DIR': r"C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP_noisy\root1-factory-20db"}
        },
        {
            'name': 'emotion2vec+ssl hfchannel 0db',
            'settings': {'USE_DACP': False, 'USE_ECDA': False, 'FIXED_CONFIDENCE_THRESHOLD': 0.75, 
                         'NOISY_DATA_DIR': r"C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP_noisy\root1-hfchannel-0db"}
        },
        {
            'name': 'emotion2vec+ssl hfchannel 10db',
            'settings': {'USE_DACP': False, 'USE_ECDA': False, 'FIXED_CONFIDENCE_THRESHOLD': 0.75, 
                         'NOISY_DATA_DIR': r"C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP_noisy\root1-hfchannel-10db"}
        },
        {
            'name': 'emotion2vec+ssl hfchannel 15db',
            'settings': {'USE_DACP': False, 'USE_ECDA': False, 'FIXED_CONFIDENCE_THRESHOLD': 0.75, 
                         'NOISY_DATA_DIR': r"C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP_noisy\root1-hfchannel-15db"}
        },
        {
            'name': 'emotion2vec+ssl hfchannel 20db',
            'settings': {'USE_DACP': False, 'USE_ECDA': False, 'FIXED_CONFIDENCE_THRESHOLD': 0.75, 
                         'NOISY_DATA_DIR': r"C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP_noisy\root1-hfchannel-20db"}
        },
        {
            'name': 'emotion2vec+ssl volvo 0db',
            'settings': {'USE_DACP': False, 'USE_ECDA': False, 'FIXED_CONFIDENCE_THRESHOLD': 0.75, 
                         'NOISY_DATA_DIR': r"C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP_noisy\root1-volvo-0db"}
        },
        {
            'name': 'emotion2vec+ssl volvo 10db',
            'settings': {'USE_DACP': False, 'USE_ECDA': False, 'FIXED_CONFIDENCE_THRESHOLD': 0.75, 
                         'NOISY_DATA_DIR': r"C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP_noisy\root1-volvo-10db"}
        },
        {
            'name': 'emotion2vec+ssl volvo 15db',
            'settings': {'USE_DACP': False, 'USE_ECDA': False, 'FIXED_CONFIDENCE_THRESHOLD': 0.75, 
                         'NOISY_DATA_DIR': r"C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP_noisy\root1-volvo-15db"}
        },
        {
            'name': 'emotion2vec+ssl volvo 20db',
            'settings': {'USE_DACP': False, 'USE_ECDA': False, 'FIXED_CONFIDENCE_THRESHOLD': 0.75, 
                         'NOISY_DATA_DIR': r"C:\Users\admin\Desktop\DATA\processed_features_IEMOCAP_noisy\root1-volvo-20db"}
        },
        
        #     'name': 'without ECDA',
        #     'settings': {'USE_DACP': True, 'USE_ECDA': False}
        # },
        # {
        #     'name': 'Baseline + ECDA',
        #     'settings': {'USE_DACP': False, 'USE_ECDA': True, 'FIXED_CONFIDENCE_THRESHOLD': 0.75}
        # }
    ]
    
    # ecda_ablations = [
    #     {
    #         'name': 'w_o Compactness Loss',
    #         'settings': {'USE_DACP': True, 'USE_ECDA': True, 'ECDA_COMPACTNESS_WEIGHT_GAMMA': 0.0, 'ECDA_REPULSION_WEIGHT_DELTA': 0.05}
    #     },
    #     {
    #         'name': 'w_o Repulsion Loss',
    #         'settings': {'USE_DACP': True, 'USE_ECDA': True, 'ECDA_COMPACTNESS_WEIGHT_GAMMA': 0.05, 'ECDA_REPULSION_WEIGHT_DELTA': 0.0}
    #     }
    # ]

    all_experiments = core_ablations
    all_results = {}
    
    for exp in all_experiments:
        all_results[exp['name']] = run_single_experiment(exp['name'], exp['settings'], fold=TARGET_FOLD)
        
    # --- 整理并打印表格 ---
    logger.info("\n" + "="*50)
    logger.info("🎉 所有IEMOCAP消融实验已完成! 🎉")
    logger.info("="*50 + "\n")

    # 手动添加完整模型结果的占位符
    full_model_placeholder = {'WA (%)': '请手动填写', 'W-F1 (%)': '请手动填写'}
    
    # 表格1: 核心模块消融
    core_table_data = {
        '模型配置': ['Baseline (CE + Consistency)', 'Baseline + DACP', 'Baseline + ECDA', '完整模型 (DACP + ECDA)'],
        'WA (%)': [
            all_results.get('Baseline (CE + Consistency)', {'WA (%)': '请手动填写'})['WA (%)'],
            all_results.get('Baseline + DACP', {'WA (%)': '请手动填写'})['WA (%)'],
            all_results.get('Baseline + ECDA', {'WA (%)': '请手动填写'})['WA (%)'],
            full_model_placeholder['WA (%)']
        ],
        'W-F1 (%)': [
            all_results.get('Baseline (CE + Consistency)', {'W-F1 (%)': '请手动填写'})['W-F1 (%)'],
            all_results.get('Baseline + DACP', {'W-F1 (%)': '请手动填写'})['W-F1 (%)'],
            all_results.get('Baseline + ECDA', {'W-F1 (%)': '请手动填写'})['W-F1 (%)'],
            full_model_placeholder['W-F1 (%)']
        ]
    }
    core_df = pd.DataFrame(core_table_data).set_index('模型配置')
    
    print("表1: IEMOCAP核心模块消融实验")
    print(core_df.to_markdown(floatfmt=".2f"))
    print("\n" + "-"*50 + "\n")

    # 表格2: ECDA内部模块消融
    ecda_table_data = {
        '模型配置': ['完整模型 (DACP + ECDA)', 'w_o Compactness Loss', 'w_o Repulsion Loss'],
        'WA (%)': [full_model_placeholder['WA (%)'], all_results['w_o Compactness Loss']['WA (%)'], all_results['w_o Repulsion Loss']['WA (%)']],
        'W-F1 (%)': [full_model_placeholder['W-F1 (%)'], all_results['w_o Compactness Loss']['W-F1 (%)'], all_results['w_o Repulsion Loss']['W-F1 (%)']]
    }
    ecda_df = pd.DataFrame(ecda_table_data).set_index('模型配置')
    
    print("表2: IEMOCAP ECDA内部模块消融实验")
    print(ecda_df.to_markdown(floatfmt=".2f"))
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    # 确保脚本从其所在目录运行，以处理相对路径问题
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main() 