#!/usr/bin/env python3
"""
快速噪声注入测试工具
用于验证噪声注入是否真的生效
"""

import argparse
import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from add_real_noise_to_audio import load_noise_files, add_real_noise


def quick_test_noise_injection():
    """快速测试噪声注入功能"""
    
    # 配置参数
    casia_root = r"C:\Users\admin\Desktop\DATA\CASIA\CASIA情感语料库"
    noise_dir = r"C:\Users\admin\Desktop\DATA\noise-92\NOISE\data\signals\5types"
    noise_type = "babble"
    snr_db = 20
    
    print("=" * 60)
    print("Quick Noise Injection Test")
    print("=" * 60)
    print(f"CASIA Root: {casia_root}")
    print(f"Noise Dir: {noise_dir}")
    print(f"Noise Type: {noise_type}")
    print(f"SNR: {snr_db} dB")
    print()
    
    # 检查目录
    if not os.path.exists(casia_root):
        print(f"❌ CASIA directory not found: {casia_root}")
        return False
    
    if not os.path.exists(noise_dir):
        print(f"❌ Noise directory not found: {noise_dir}")
        return False
    
    # 加载噪声文件
    print("Loading noise files...")
    noise_files = load_noise_files(noise_dir)
    
    if noise_type not in noise_files:
        print(f"❌ Noise type '{noise_type}' not found")
        print(f"Available types: {list(noise_files.keys())}")
        return False
    
    print(f"✅ Found {len(noise_files[noise_type])} noise files for type '{noise_type}'")
    
    # 寻找一个测试音频文件
    test_audio_path = None
    for root, dirs, files in os.walk(casia_root):
        for file in files:
            if file.lower().endswith('.wav'):
                test_audio_path = os.path.join(root, file)
                break
        if test_audio_path:
            break
    
    if not test_audio_path:
        print("❌ No WAV files found in CASIA directory")
        return False
    
    print(f"✅ Using test audio: {os.path.relpath(test_audio_path, casia_root)}")
    
    # 读取原始音频
    print("Reading original audio...")
    try:
        original_audio, sr_original = sf.read(test_audio_path)
        print(f"  Duration: {len(original_audio)/sr_original:.2f}s")
        print(f"  Sample rate: {sr_original} Hz")
        print(f"  RMS: {np.sqrt(np.mean(original_audio**2)):.4f}")
        print(f"  Peak: {np.max(np.abs(original_audio)):.4f}")
    except Exception as e:
        print(f"❌ Error reading original audio: {e}")
        return False
    
    # 读取噪声文件
    print(f"Reading {noise_type} noise...")
    try:
        noise_file = noise_files[noise_type][0]  # 使用第一个噪声文件
        noise_audio, sr_noise = sf.read(noise_file)
        print(f"  Noise file: {os.path.basename(noise_file)}")
        print(f"  Duration: {len(noise_audio)/sr_noise:.2f}s") 
        print(f"  Sample rate: {sr_noise} Hz")
        print(f"  RMS: {np.sqrt(np.mean(noise_audio**2)):.4f}")
        print(f"  Peak: {np.max(np.abs(noise_audio)):.4f}")
    except Exception as e:
        print(f"❌ Error reading noise audio: {e}")
        return False
    
    # 应用噪声
    print(f"Applying {noise_type} noise at {snr_db} dB SNR...")
    try:
        noisy_audio = add_real_noise(original_audio, noise_audio, snr_db, sr_original, sr_noise)
        print(f"  Noisy RMS: {np.sqrt(np.mean(noisy_audio**2)):.4f}")
        print(f"  Noisy Peak: {np.max(np.abs(noisy_audio)):.4f}")
        
        # 计算变化
        rms_change = (np.sqrt(np.mean(noisy_audio**2)) / np.sqrt(np.mean(original_audio**2)) - 1) * 100
        peak_change = (np.max(np.abs(noisy_audio)) / np.max(np.abs(original_audio)) - 1) * 100
        
        print(f"  RMS change: {rms_change:+.1f}%")
        print(f"  Peak change: {peak_change:+.1f}%")
        
    except Exception as e:
        print(f"❌ Error applying noise: {e}")
        return False
    
    # 保存测试文件
    print("Saving test files...")
    test_dir = "noise_test_output"
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # 保存原始音频
        orig_path = os.path.join(test_dir, "original.wav")
        sf.write(orig_path, original_audio, sr_original)
        print(f"  Original: {orig_path}")
        
        # 保存噪声音频  
        noisy_path = os.path.join(test_dir, f"noisy_{noise_type}_{snr_db}db.wav")
        sf.write(noisy_path, noisy_audio, sr_original)
        print(f"  Noisy: {noisy_path}")
        
        # 保存纯噪声（用于参考）
        if len(noise_audio) > len(original_audio):
            noise_segment = noise_audio[:len(original_audio)]
        else:
            repeat_times = (len(original_audio) // len(noise_audio)) + 1
            noise_segment = np.tile(noise_audio, repeat_times)[:len(original_audio)]
        
        noise_path = os.path.join(test_dir, f"pure_{noise_type}.wav")
        sf.write(noise_path, noise_segment, sr_original)
        print(f"  Pure noise: {noise_path}")
        
    except Exception as e:
        print(f"❌ Error saving files: {e}")
        return False
    
    # 验证结果
    print("\n" + "=" * 40)
    print("VERIFICATION")
    print("=" * 40)
    
    # 检查是否有实际变化
    if np.allclose(original_audio, noisy_audio, rtol=1e-10):
        print("❌ CRITICAL: Original and noisy audio are identical!")
        print("   Noise injection did NOT work!")
        return False
    else:
        print("✅ Audio files are different - noise was added")
    
    # 检查RMS变化
    if abs(rms_change) < 1:
        print("⚠️  WARNING: RMS change is very small (<1%)")
        print("   Noise level might be too low")
    else:
        print(f"✅ RMS changed by {rms_change:+.1f}% - reasonable noise level")
    
    # 估算实际SNR
    signal_power = np.mean(original_audio ** 2)
    total_power = np.mean(noisy_audio ** 2)
    noise_power = total_power - signal_power
    
    if noise_power > 0:
        actual_snr = 10 * np.log10(signal_power / noise_power)
        snr_error = abs(actual_snr - snr_db)
        print(f"✅ Estimated actual SNR: {actual_snr:.1f} dB (target: {snr_db} dB)")
        
        if snr_error > 3:
            print(f"⚠️  WARNING: SNR error is {snr_error:.1f} dB (>3dB)")
        else:
            print("✅ SNR is within acceptable range")
    else:
        print("❌ Could not calculate SNR - noise power ≤ 0")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 NOISE INJECTION TEST COMPLETED!")
    print(f"📁 Test files saved in: {os.path.abspath(test_dir)}")
    print("🎧 Listen to the files to verify the difference manually")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = quick_test_noise_injection()
    exit(0 if success else 1)