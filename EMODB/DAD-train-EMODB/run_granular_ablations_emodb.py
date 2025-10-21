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
    运行单次EMODB实验。
    :param experiment_name: 实验名称，用于创建独立的结果目录。
    :param config_overrides: 需要覆盖的配置项。
    :param fold: 交叉验证的折数。
    :return: 一个包含 'WA' 和 'W-F1' 的字典。
    """
    logger.info(f"--- 🚀 开始EMODB消融实验: {experiment_name} (Fold {fold+1}) ---")
    
    # --- 动态加载和修改配置 ---
    import config_emodb as config
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
    import train_emodb as train
    importlib.reload(train)
    
    trainer = train.FixedEMODBCrossDomainTrainer(fold=fold, experiment_name=experiment_name)
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

def run_experiment_on_multiple_noises(experiment_name, base_settings, noise_types, fold):
    """
    在多种噪声类型上运行单个实验，并计算平均性能
    :param experiment_name: 实验基础名称
    :param base_settings: 基础配置设置
    :param noise_types: 噪声类型字典 {'noise_name': 'path'}
    :param fold: 交叉验证折数
    :return: 平均结果字典
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"🧪 开始多噪声实验: {experiment_name}")
    logger.info(f"    📊 将在 {len(noise_types)} 种噪声类型上测试: {list(noise_types.keys())}")
    logger.info(f"{'='*70}")
    
    noise_results = {}
    
    for noise_name, noise_path in noise_types.items():
        logger.info(f"\n--- 🔊 测试噪声类型: {noise_name} ---")
        
        # 为每种噪声类型创建完整的配置
        full_settings = base_settings.copy()
        full_settings['NOISY_DATA_DIR'] = noise_path
        
        # 创建带噪声类型的实验名称
        full_experiment_name = f"{experiment_name}_{noise_name}_10db"
        
        try:
            result = run_single_experiment(full_experiment_name, full_settings, fold)
            noise_results[noise_name] = result
            logger.info(f"✅ {noise_name} 完成 - WA: {result['WA (%)']}%, W-F1: {result['W-F1 (%)']}%")
        except Exception as e:
            logger.error(f"❌ {noise_name} 失败: {str(e)}")
            noise_results[noise_name] = {'WA (%)': 'Error', 'W-F1 (%)': 'Error'}
    
    # 计算有效结果的平均值
    valid_wa_results = []
    valid_f1_results = []
    
    for noise_name, result in noise_results.items():
        if isinstance(result['WA (%)'], (int, float)):
            valid_wa_results.append(result['WA (%)'])
        if isinstance(result['W-F1 (%)'], (int, float)):
            valid_f1_results.append(result['W-F1 (%)'])
    
    if valid_wa_results and valid_f1_results:
        avg_wa = sum(valid_wa_results) / len(valid_wa_results)
        avg_f1 = sum(valid_f1_results) / len(valid_f1_results)
        
        logger.info(f"\n📈 {experiment_name} 平均性能:")
        logger.info(f"    🎯 平均WA: {avg_wa:.2f}% (基于 {len(valid_wa_results)}/{len(noise_types)} 个有效结果)")
        logger.info(f"    🎯 平均W-F1: {avg_f1:.2f}% (基于 {len(valid_f1_results)}/{len(noise_types)} 个有效结果)")
        
        return {
            'WA (%)': avg_wa,
            'W-F1 (%)': avg_f1,
            'individual_results': noise_results,
            'valid_count': len(valid_wa_results)
        }
    else:
        logger.error(f"❌ {experiment_name} 无有效结果")
        return {
            'WA (%)': 'Error',
            'W-F1 (%)': 'Error',
            'individual_results': noise_results,
            'valid_count': 0
        }

def main():
    """主函数，定义并运行所有EMODB细粒度消融实验"""
    
    # --- 实验配置 ---
    # EMODB 使用第1折 (fold=0)，测试四种噪声类型的10dB条件
    TARGET_FOLD = 3  # EMODB默认使用第1折
    BASE_NOISE_PATH = r"C:\Users\admin\Desktop\DATA\processed_features_EMODB_noisy"
    
    # 四种噪声类型的10dB数据路径
    NOISE_TYPES = {
        'babble': f"{BASE_NOISE_PATH}\\root1-babble-10db",
        # 'factory': f"{BASE_NOISE_PATH}\\root1-factory-10db", 
        # 'hfchannel': f"{BASE_NOISE_PATH}\\root1-hfchannel-10db",
        # 'volvo': f"{BASE_NOISE_PATH}\\root1-volvo-10db"
    }

    # 定义所有消融实验（不包含噪声路径，运行时动态设置）
    granular_ablations = [
        # 1. 完整模型（基准）
        # {
        #     'name': 'Proposed_Full_Model',
        #     'settings': {
        #         'USE_DACP': True, 
        #         'USE_ECDA': True,
        #         'ANCHOR_CALIBRATION_ENABLED': True,
        #         'USE_ENTROPY_IN_SCORE': True,
        #         'USE_CLASS_AWARE_MMD': True,
        #         'ECDA_CLASS_ATTENTION_LAMBDA': 1.0,
        #         'ECDA_COMPACTNESS_WEIGHT_GAMMA': 0.1,
        #         'ECDA_REPULSION_WEIGHT_DELTA': 0.1
        #     }
        # },
        
        # 2. 基线模型（无DACP无ECDA）
        {
            'name': 'Baseline_No_DACP_No_ECDA',
            'settings': {
                'USE_DACP': False, 
                'USE_ECDA': False,
                'FIXED_CONFIDENCE_THRESHOLD': 0.75
            }
        },
        
        # 3. DACP消融：无锚点校准
        {
            'name': 'Ablation_DACP_No_Anchor',
            'settings': {
                'USE_DACP': True, 
                'USE_ECDA': True,
                'ANCHOR_CALIBRATION_ENABLED': False
            }
        },
        
        # 4. DACP消融：无类别自适应
        {
            'name': 'Ablation_DACP_No_ClassAdapt',
            'settings': {
                'USE_DACP': True, 
                'USE_ECDA': True,
                'DACP_SENSITIVITY_K': 0.0  # 禁用sigmoid敏感系数
            }
        },
        
        # 5. DACP消融：无课程学习
        {
            'name': 'Ablation_DACP_No_Curriculum',
            'settings': {
                'USE_DACP': True, 
                'USE_ECDA': True,
                'DACP_QUANTILE_START': 0.4,  # 设置为固定值，禁用动态变化
                'DACP_QUANTILE_END': 0.4
            }
        },
        
        # 6. DACP消融：使用简单置信度
        {
            'name': 'Ablation_DACP_Simple_Confidence',
            'settings': {
                'USE_DACP': True, 
                'USE_ECDA': True,
                'USE_ENTROPY_IN_SCORE': False  # 禁用熵增强的置信度分数
            }
        },
        
        # 7. 无ECDA（仅DACP）
        {
            'name': 'Ablation_No_ECDA_Only_DACP',
            'settings': {
                'USE_DACP': True, 
                'USE_ECDA': False
            }
        },
        
        # 8. 无DACP（仅ECDA）
        {
            'name': 'Ablation_No_DACP_Only_ECDA',
            'settings': {
                'USE_DACP': False,
                'USE_ECDA': True,
                'FIXED_CONFIDENCE_THRESHOLD': 0.75  # 不使用DACP时需要固定阈值
            }
        },
        
        # 9. ECDA消融：替换为全局MMD
        {
            'name': 'Ablation_ECDA_Global_MMD',
            'settings': {
                'USE_DACP': True, 
                'USE_ECDA': True,
                'USE_CLASS_AWARE_MMD': False,  # 使用全局MMD
                'ECDA_COMPACTNESS_WEIGHT_GAMMA': 0.0,  # 禁用紧凑性损失
                'ECDA_REPULSION_WEIGHT_DELTA': 0.0     # 禁用斥力损失
            }
        },
        
        # 10. ECDA消融：无类别级注意力
        {
            'name': 'Ablation_ECDA_No_ClassAttention',
            'settings': {
                'USE_DACP': True, 
                'USE_ECDA': True,
                'ECDA_CLASS_ATTENTION_LAMBDA': 0.0  # 禁用类别级注意力
            }
        },
        
        # 11. ECDA消融：无紧凑性损失
        {
            'name': 'Ablation_ECDA_No_Compactness',
            'settings': {
                'USE_DACP': True, 
                'USE_ECDA': True,
                'ECDA_COMPACTNESS_WEIGHT_GAMMA': 0.0  # 禁用紧凑性损失
            }
        },
        
        # 12. ECDA消融：无斥力损失
        {
            'name': 'Ablation_ECDA_No_Repulsion',
            'settings': {
                'USE_DACP': True, 
                'USE_ECDA': True,
                'ECDA_REPULSION_WEIGHT_DELTA': 0.0  # 禁用斥力损失
            }
        }
    ]
    
    # 运行所有消融实验（在四种噪声类型上）
    all_results = {}
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 开始EMODB细粒度消融实验 (Fold {TARGET_FOLD+1}, 10dB噪声) 🚀")
    logger.info(f"📊 将在 {len(NOISE_TYPES)} 种噪声类型上测试: {list(NOISE_TYPES.keys())}")
    logger.info(f"{'='*80}")
    logger.info(f"📋 计划运行 {len(granular_ablations)} 个消融实验 × {len(NOISE_TYPES)} 种噪声 = {len(granular_ablations) * len(NOISE_TYPES)} 个训练任务")
    logger.info(f"{'='*80}\n")
    
    for i, exp in enumerate(granular_ablations, 1):
        logger.info(f"\n{'='*90}")
        logger.info(f"🧪 消融实验 {i}/{len(granular_ablations)}: {exp['name']}")
        logger.info(f"{'='*90}")
        
        try:
            result = run_experiment_on_multiple_noises(
                experiment_name=exp['name'],
                base_settings=exp['settings'], 
                noise_types=NOISE_TYPES,
                fold=TARGET_FOLD
            )
            all_results[exp['name']] = result
            
            if isinstance(result['WA (%)'], (int, float)):
                logger.info(f"✅ 消融实验 {i} 完成 - 平均WA: {result['WA (%)']:.2f}%, 平均W-F1: {result['W-F1 (%)']:.2f}% (基于 {result['valid_count']}/{len(NOISE_TYPES)} 种噪声)")
            else:
                logger.error(f"❌ 消融实验 {i} 失败 - 无有效结果")
                
        except Exception as e:
            logger.error(f"❌ 消融实验 {i} 失败: {str(e)}")
            all_results[exp['name']] = {
                'WA (%)': 'Error', 
                'W-F1 (%)': 'Error',
                'individual_results': {},
                'valid_count': 0
            }
    
    # --- 整理并生成结果表格 ---
    logger.info(f"\n{'='*80}")
    logger.info("🎉 所有EMODB细粒度消融实验已完成! 🎉")
    logger.info(f"{'='*80}\n")

    # 创建分层结果表格
    table_data = []
    
    # 按逻辑顺序组织结果（移除Proposed_Full_Model，因为用户已有数据）
    experiment_order = [
        ('Ablation_DACP_No_Anchor', 'w/o Anchor Calibration', 0),
        ('Ablation_DACP_No_ClassAdapt', 'w/o Class-Adaptivity', 0),
        ('Ablation_DACP_No_Curriculum', 'w/o Curriculum Progression', 0),
        ('Ablation_DACP_Simple_Confidence', 'w/o Uncertainty Score', 0),
        ('Ablation_No_ECDA_Only_DACP', 'w/o ECDA (Only DACP)', 0),
        ('Ablation_No_DACP_Only_ECDA', 'w/o DACP (Only ECDA)', 0),
        ('Ablation_ECDA_Global_MMD', 'ECDA w/ Global MMD', 0),
        ('Ablation_ECDA_No_ClassAttention', 'ECDA w/o Class Attention', 0),
        ('Ablation_ECDA_No_Compactness', 'ECDA w/o Compactness', 0),
        ('Ablation_ECDA_No_Repulsion', 'ECDA w/o Repulsion', 0),
        ('Baseline_No_DACP_No_ECDA', 'Baseline (No DACP & No ECDA)', 0)
    ]
    
    for exp_key, display_name, level in experiment_order:
        if exp_key in all_results:
            result = all_results[exp_key]
            # 提取平均性能（如果是数值）或错误信息
            wa_avg = result['WA (%)'] if isinstance(result['WA (%)'], (int, float)) else result['WA (%)']
            f1_avg = result['W-F1 (%)'] if isinstance(result['W-F1 (%)'], (int, float)) else result['W-F1 (%)']
            
            table_data.append({
                'Approach': display_name,
                'WA (%)': wa_avg,
                'W-F1 (%)': f1_avg
            })
        else:
            table_data.append({
                'Approach': display_name,
                'WA (%)': 'N/A',
                'W-F1 (%)': 'N/A'
            })
    
    # 创建DataFrame并输出
    results_df = pd.DataFrame(table_data)
    
    print("\n" + "="*80)
    print("📊 EMODB细粒度消融实验结果汇总")
    print(f"   🔊 四种噪声类型10dB平均性能: {list(NOISE_TYPES.keys())}")
    print("="*80)
    print(results_df.to_markdown(index=False, floatfmt=".2f"))
    print("="*80)
    
    # 保存结果到CSV文件
    csv_path = f'emodb_granular_ablation_results_fold_{TARGET_FOLD+1}_10db.csv'
    results_df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"\n💾 结果已保存至: {csv_path}")
    
    # 生成详细的噪声类型分解表格
    print(f"\n📋 详细噪声类型分解表格:")
    print("="*100)
    detailed_data = []
    
    for exp_key, display_name, level in experiment_order:
        if exp_key in all_results and 'individual_results' in all_results[exp_key]:
            row_data = {'Approach': display_name}
            individual_results = all_results[exp_key]['individual_results']
            
            # 添加各种噪声类型的结果
            for noise_name in NOISE_TYPES.keys():
                if noise_name in individual_results:
                    wa_val = individual_results[noise_name]['WA (%)']
                    f1_val = individual_results[noise_name]['W-F1 (%)']
                    row_data[f'{noise_name}_WA'] = wa_val if isinstance(wa_val, (int, float)) else 'Error'
                    row_data[f'{noise_name}_F1'] = f1_val if isinstance(f1_val, (int, float)) else 'Error'
                else:
                    row_data[f'{noise_name}_WA'] = 'N/A'
                    row_data[f'{noise_name}_F1'] = 'N/A'
            
            # 添加平均值
            avg_wa = all_results[exp_key]['WA (%)']
            avg_f1 = all_results[exp_key]['W-F1 (%)']
            row_data['Avg_WA'] = avg_wa if isinstance(avg_wa, (int, float)) else 'Error'
            row_data['Avg_F1'] = avg_f1 if isinstance(avg_f1, (int, float)) else 'Error'
            
            detailed_data.append(row_data)
    
    if detailed_data:
        detailed_df = pd.DataFrame(detailed_data)
        print(detailed_df.to_markdown(index=False, floatfmt=".2f"))
        
        # 保存详细表格
        detailed_csv_path = f'emodb_detailed_noise_breakdown_fold_{TARGET_FOLD+1}_10db.csv'
        detailed_df.to_csv(detailed_csv_path, index=False, encoding='utf-8')
        logger.info(f"💾 详细分解表格已保存至: {detailed_csv_path}")

    # 注意：完整模型性能分析已移除，因为用户已有完整模型数据
    print(f"\n📝 注意: Proposed_Full_Model 未包含在此次实验中")
    print(f"         请使用您之前的完整模型数据作为基线进行性能对比分析")
    
    print("\n" + "="*80)
    print("✅ 消融实验分析完成！")
    print("📊 主要结果表格和详细分解表格都已生成，请复制到您的论文中。")
    print("="*80)

if __name__ == "__main__":
    # 确保脚本从其所在目录运行，以处理相对路径问题
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()