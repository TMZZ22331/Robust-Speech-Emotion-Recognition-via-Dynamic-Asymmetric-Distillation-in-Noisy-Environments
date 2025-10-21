#!/usr/bin/env python3
"""
测试不同SNR级别的噪声效果
"""

import os
import numpy as np  
import soundfile as sf
from add_real_noise_to_audio import load_noise_files, add_real_noise


def test_multiple_snr_levels():
    """测试多个SNR级别"""
    
    # 配置
    casia_root = r"C:\Users\admin\Desktop\DATA\CASIA\CASIA情感语料库"
    noise_dir = r"C:\Users\admin\Desktop\DATA\noise-92\NOISE\data\signals\5types"
    noise_type = "babble"
    
    # 测试的SNR级别
    snr_levels = [0, 5, 10, 15, 20]
    
    print("=" * 70)
    print("Multi-SNR Noise Injection Test")
    print("=" * 70)
    
    # 找一个测试音频
    test_audio_path = None
    for root, dirs, files in os.walk(casia_root):
        for file in files:
            if file.lower().endswith('.wav'):
                test_audio_path = os.path.join(root, file)
                break
        if test_audio_path:
            break
    
    if not test_audio_path:
        print("❌ No test audio found")
        return
    
    # 加载音频和噪声
    original_audio, sr = sf.read(test_audio_path)
    noise_files = load_noise_files(noise_dir)
    noise_audio, sr_noise = sf.read(noise_files[noise_type][0])
    
    print(f"Test audio: {os.path.relpath(test_audio_path, casia_root)}")
    print(f"Original RMS: {np.sqrt(np.mean(original_audio**2)):.4f}")
    print()
    
    results = []
    
    for snr_db in snr_levels:
        print(f"Testing SNR: {snr_db} dB")
        
        # 应用噪声
        noisy_audio = add_real_noise(original_audio, noise_audio, snr_db, sr, sr_noise)
        
        # 计算统计数据
        orig_rms = np.sqrt(np.mean(original_audio**2))
        noisy_rms = np.sqrt(np.mean(noisy_audio**2))
        rms_change = (noisy_rms / orig_rms - 1) * 100
        
        # 估算实际SNR
        signal_power = np.mean(original_audio ** 2)
        total_power = np.mean(noisy_audio ** 2)
        noise_power = total_power - signal_power
        
        if noise_power > 0:
            actual_snr = 10 * np.log10(signal_power / noise_power)
        else:
            actual_snr = float('inf')
        
        # 检查是否有变化
        has_change = not np.allclose(original_audio, noisy_audio, rtol=1e-10)
        
        results.append({
            'target_snr': snr_db,
            'actual_snr': actual_snr,
            'rms_change': rms_change,
            'has_change': has_change
        })
        
        print(f"  RMS change: {rms_change:+6.1f}%")
        print(f"  Actual SNR: {actual_snr:6.1f} dB")
        print(f"  Has change: {'✅' if has_change else '❌'}")
        print()
    
    # 总结结果
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("SNR(dB) | RMS Change | Actual SNR | Status")
    print("-" * 70)
    
    for result in results:
        status = "✅ Good" if result['has_change'] and abs(result['rms_change']) > 5 else "⚠️ Weak" if result['has_change'] else "❌ None"
        print(f"{result['target_snr']:6d} | {result['rms_change']:+9.1f}% | {result['actual_snr']:9.1f} | {status}")
    
    print()
    print("Recommendations:")
    strong_snr = [r for r in results if r['has_change'] and abs(r['rms_change']) > 10]
    if strong_snr:
        best_snr = min(strong_snr, key=lambda x: x['target_snr'])['target_snr']
        print(f"✅ Use SNR ≤ {best_snr} dB for significant noise impact")
    else:
        print("⚠️ Consider using even lower SNR levels (negative values)")
    
    print(f"📁 For comparison, original 20dB had {results[-1]['rms_change']:+.1f}% RMS change")


if __name__ == "__main__":
    test_multiple_snr_levels()