#!/usr/bin/env python3
"""
快速测试脚本：生成分析数据
运行短时间训练以生成 training_history.json 和 confirmation_bias_log.json
"""

import os
import sys
import torch

# 临时修改配置以进行快速测试
import config as cfg

# 备份原始值
original_epochs = cfg.EPOCHS
original_warmup = cfg.WARMUP_EPOCHS
original_early_stopping = cfg.EARLY_STOPPING

# 设置快速测试参数
cfg.EPOCHS = 150  # 运行150个epoch (30个warmup + 120个正常训练)
cfg.WARMUP_EPOCHS = 30  # 保持原始warmup epochs
cfg.EARLY_STOPPING = False  # 禁用早停以确保运行完整
cfg.VALIDATION_INTERVAL = 5  # 每5epoch验证一次

print("🧪 快速测试配置:")
print(f"   - 总轮次: {cfg.EPOCHS}")
print(f"   - 预热轮次: {cfg.WARMUP_EPOCHS}")
print(f"   - 验证间隔: {cfg.VALIDATION_INTERVAL}")
print(f"   - 早停: {cfg.EARLY_STOPPING}")

# 导入训练器
from train import IEMOCAPCrossDomainTrainer

def main():
    print("🚀 开始快速测试训练以生成分析数据...")
    
    try:
        # 创建训练器
        trainer = IEMOCAPCrossDomainTrainer(
            fold=cfg.N_FOLDS-1, 
            experiment_name="Analysis_Data_Test"
        )
        
        # 运行训练
        results = trainer.train()
        
        print("\n✅ 快速测试训练完成!")
        print(f"📊 最佳噪声域加权准确率: {results['best_noisy_weighted_acc']:.2f}%")
        print(f"📁 结果保存目录: {results['results_dir']}")
        
        # 检查生成的分析文件
        reports_dir = os.path.join(results['results_dir'], "reports")
        history_file = os.path.join(reports_dir, "training_history.json")
        bias_file = os.path.join(reports_dir, "confirmation_bias_log.json")
        
        print("\n📋 生成的分析文件:")
        if os.path.exists(history_file):
            print(f"   ✅ training_history.json ({os.path.getsize(history_file):,} bytes)")
        else:
            print(f"   ❌ training_history.json 未生成")
            
        if os.path.exists(bias_file):
            print(f"   ✅ confirmation_bias_log.json ({os.path.getsize(bias_file):,} bytes)")
        else:
            print(f"   ❌ confirmation_bias_log.json 未生成")
        
        return results['results_dir']
        
    except Exception as e:
        print(f"❌ 测试训练失败: {e}")
        raise
    finally:
        # 恢复原始配置
        cfg.EPOCHS = original_epochs
        cfg.WARMUP_EPOCHS = original_warmup  
        cfg.EARLY_STOPPING = original_early_stopping
        print("\n🔄 已恢复原始配置参数")

if __name__ == "__main__":
    main() 