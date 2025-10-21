#!/usr/bin/env python3
"""
噪声注入验证脚本
比较干净音频和噪声音频，确保噪声真的被注入了
"""

import argparse
import os
import numpy as np
import soundfile as sf
import glob
import random
from pathlib import Path


def load_audio_safely(file_path):
    """安全加载音频文件"""
    try:
        audio, sr = sf.read(file_path)
        
        # 如果是立体声，转换为单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        return audio, sr
    except Exception as e:
        print(f"[ERROR] 加载音频文件失败 {file_path}: {e}")
        return None, None


def calculate_audio_stats(audio):
    """计算音频统计特性"""
    if audio is None or len(audio) == 0:
        return None
    
    return {
        'mean': float(np.mean(audio)),
        'std': float(np.std(audio)),
        'rms': float(np.sqrt(np.mean(audio ** 2))),
        'max_amplitude': float(np.max(np.abs(audio))),
        'energy': float(np.sum(audio ** 2)),
        'length': len(audio)
    }


def calculate_snr(clean_audio, noisy_audio):
    """计算实际的信噪比"""
    if clean_audio is None or noisy_audio is None:
        return None
    
    if len(clean_audio) != len(noisy_audio):
        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]
    
    # 估算噪声信号（假设噪声是加性的）
    noise_estimate = noisy_audio - clean_audio
    
    # 计算信号和噪声功率
    signal_power = np.mean(clean_audio ** 2)
    noise_power = np.mean(noise_estimate ** 2)
    
    if noise_power == 0:
        return float('inf')  # 完全没有噪声
    
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else -float('inf')
    
    return snr_db


def verify_noise_injection(clean_root, noisy_root, expected_snr_db, noise_type, 
                         sample_count=10, tolerance_db=3.0):
    """
    验证噪声注入效果
    
    Args:
        clean_root: 干净音频根目录
        noisy_root: 噪声音频根目录  
        expected_snr_db: 期望的SNR (dB)
        noise_type: 噪声类型
        sample_count: 验证的样本数量
        tolerance_db: SNR容差 (dB)
    
    Returns:
        bool: 验证是否通过
    """
    
    print(f"[VERIFY] 验证噪声注入效果...")
    print(f"   干净音频目录: {clean_root}")
    print(f"   噪声音频目录: {noisy_root}")
    print(f"   期望SNR: {expected_snr_db} dB")
    print(f"   噪声类型: {noise_type}")
    print(f"   验证样本数: {sample_count}")
    
    # 检查目录是否存在
    if not os.path.exists(clean_root):
        print(f"[ERROR] 干净音频目录不存在: {clean_root}")
        return False
    
    if not os.path.exists(noisy_root):
        print(f"[ERROR] 噪声音频目录不存在: {noisy_root}")
        return False
    
    # 查找音频文件
    clean_files = glob.glob(os.path.join(clean_root, "**", "*.wav"), recursive=True)
    noisy_files = glob.glob(os.path.join(noisy_root, "**", "*.wav"), recursive=True)
    
    if not clean_files:
        print(f"[ERROR] 在干净音频目录中未找到.wav文件")
        return False
    
    if not noisy_files:
        print(f"[ERROR] 在噪声音频目录中未找到.wav文件")
        return False
    
    print(f"[OK] 找到干净音频文件: {len(clean_files)} 个")
    print(f"[OK] 找到噪声音频文件: {len(noisy_files)} 个")
    
    # 创建噪声文件映射 (相对路径 -> 噪声文件路径)
    noisy_file_map = {}
    for noisy_file in noisy_files:
        rel_path = os.path.relpath(noisy_file, noisy_root)
        noisy_file_map[rel_path] = noisy_file
    
    # 筛选出有对应噪声文件的干净文件
    available_clean_files = []
    for clean_file in clean_files:
        rel_path = os.path.relpath(clean_file, clean_root)
        if rel_path in noisy_file_map:
            available_clean_files.append(clean_file)
    
    if not available_clean_files:
        print("[ERROR] 没有找到有对应噪声文件的干净音频文件")
        return False
    
    print(f"[OK] 找到可配对的音频文件: {len(available_clean_files)} 个")
    
    # 随机选择样本进行验证
    verification_count = min(sample_count, len(available_clean_files))
    selected_clean_files = random.sample(available_clean_files, verification_count)
    
    verified_samples = 0
    total_snr_error = 0.0
    snr_measurements = []
    significant_differences = 0
    
    print(f"\n[INFO] 开始验证 {verification_count} 个音频样本...")
    
    for i, clean_file in enumerate(selected_clean_files):
        # 使用映射表查找对应的噪声文件
        rel_path = os.path.relpath(clean_file, clean_root)
        noisy_file = noisy_file_map.get(rel_path)
        
        if not noisy_file or not os.path.exists(noisy_file):
            print(f"[WARN] 样本 {i+1}: 找不到对应的噪声文件 {rel_path}")
            continue
        
        # 加载音频
        clean_audio, clean_sr = load_audio_safely(clean_file)
        noisy_audio, noisy_sr = load_audio_safely(noisy_file)
        
        if clean_audio is None or noisy_audio is None:
            print(f"[WARN] 样本 {i+1}: 音频加载失败")
            continue
        
        if clean_sr != noisy_sr:
            print(f"[WARN] 样本 {i+1}: 采样率不匹配 ({clean_sr} vs {noisy_sr})")
            continue
        
        # 计算统计特性
        clean_stats = calculate_audio_stats(clean_audio)
        noisy_stats = calculate_audio_stats(noisy_audio)
        
        # 检查是否有显著差异
        rms_diff = abs(noisy_stats['rms'] - clean_stats['rms'])
        energy_ratio = noisy_stats['energy'] / clean_stats['energy'] if clean_stats['energy'] > 0 else 1.0
        
        # 计算实际SNR
        actual_snr = calculate_snr(clean_audio, noisy_audio)
        
        if actual_snr is not None and not np.isinf(actual_snr):
            snr_error = abs(actual_snr - expected_snr_db)
            snr_measurements.append(actual_snr)
            total_snr_error += snr_error
            
            print(f"   样本 {i+1}: SNR={actual_snr:.2f}dB (误差: {snr_error:.2f}dB)")
            
            if snr_error <= tolerance_db:
                verified_samples += 1
        else:
            print(f"   样本 {i+1}: SNR计算失败")
        
        # 检查是否有显著的音频差异 (放宽阈值)
        if rms_diff > 0.0001 or abs(energy_ratio - 1.0) > 0.01:
            significant_differences += 1
            print(f"     [OK] 检测到音频差异: RMS差异={rms_diff:.6f}, 能量比={energy_ratio:.4f}")
    
    # 验证结果分析
    print(f"\n[RESULT] 验证结果:")
    print(f"   成功验证样本: {verified_samples}/{verification_count}")
    print(f"   检测到显著差异: {significant_differences}/{verification_count}")
    
    if snr_measurements:
        avg_snr = np.mean(snr_measurements)
        avg_error = total_snr_error / len(snr_measurements)
        print(f"   平均SNR: {avg_snr:.2f} dB (期望: {expected_snr_db} dB)")
        print(f"   平均误差: {avg_error:.2f} dB")
    
    # 判断验证是否通过
    success_rate = verified_samples / verification_count if verification_count > 0 else 0
    difference_rate = significant_differences / verification_count if verification_count > 0 else 0
    
    # 验证标准 (调整为更合理的阈值)
    min_success_rate = 0.6  # 至少60%的样本SNR误差在容差范围内
    min_difference_rate = 0.5  # 至少50%的样本有显著差异
    
    print(f"\n[CRITERIA] 验证标准:")
    print(f"   SNR准确率: {success_rate:.2%} (要求: ≥{min_success_rate:.0%})")
    print(f"   差异检出率: {difference_rate:.2%} (要求: ≥{min_difference_rate:.0%})")
    
    if success_rate >= min_success_rate and difference_rate >= min_difference_rate:
        print(f"\n[OK] 噪声注入验证通过!")
        print(f"   {noise_type} 噪声已成功注入，SNR={expected_snr_db}dB")
        return True
    else:
        print(f"\n[ERROR] 噪声注入验证失败!")
        if success_rate < min_success_rate:
            print(f"   SNR准确率不足: {success_rate:.2%} < {min_success_rate:.0%}")
        if difference_rate < min_difference_rate:
            print(f"   未检测到足够的音频差异: {difference_rate:.2%} < {min_difference_rate:.0%}")
        print(f"   可能原因:")
        print(f"   1. 噪声没有被正确添加")
        print(f"   2. SNR设置不正确")
        print(f"   3. 音频处理出现问题")
        return False


def get_parser():
    parser = argparse.ArgumentParser(description="验证噪声注入效果")
    parser.add_argument(
        "--clean_root",
        required=True,
        help="干净音频根目录"
    )
    parser.add_argument(
        "--noisy_root", 
        required=True,
        help="噪声音频根目录"
    )
    parser.add_argument(
        "--expected_snr",
        type=float,
        required=True,
        help="期望的SNR (dB)"
    )
    parser.add_argument(
        "--noise_type",
        required=True,
        help="噪声类型"
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=10,
        help="验证的样本数量 (默认: 10)"
    )
    parser.add_argument(
        "--tolerance",
        type=float, 
        default=3.0,
        help="SNR容差 (dB) (默认: 3.0)"
    )
    return parser


def main(args):
    print("=" * 60)
    print("[TEST] 噪声注入验证工具")
    print("=" * 60)
    
    # 设置随机种子保证可重现性
    np.random.seed(42)
    random.seed(42)
    
    success = verify_noise_injection(
        clean_root=args.clean_root,
        noisy_root=args.noisy_root,
        expected_snr_db=args.expected_snr,
        noise_type=args.noise_type,
        sample_count=args.sample_count,
        tolerance_db=args.tolerance
    )
    
    print("=" * 60)
    
    if success:
        print("[SUCCESS] 验证成功! 噪声注入正常工作。")
        return 0
    else:
        print("[ERROR] 验证失败! 噪声注入存在问题。")
        return 1


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    exit(main(args)) 