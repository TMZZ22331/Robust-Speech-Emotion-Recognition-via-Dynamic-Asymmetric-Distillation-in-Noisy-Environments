#!/usr/bin/env python3
"""
验证真实噪声注入脚本
用于测试真实噪声注入是否正常工作
"""

import argparse
import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_audio_properties(audio_path):
    """分析音频文件的基本属性"""
    try:
        audio, sr = sf.read(audio_path)
        
        # 基本统计信息
        duration = len(audio) / sr
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        
        # 频谱分析
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(fft), 1/sr)
        magnitude = np.abs(fft)
        
        return {
            'duration': duration,
            'sample_rate': sr,
            'rms': rms,
            'peak': peak,
            'length': len(audio),
            'freqs': freqs[:len(freqs)//2],
            'magnitude': magnitude[:len(magnitude)//2]
        }
    except Exception as e:
        print(f"Error analyzing {audio_path}: {e}")
        return None


def plot_comparison(original_props, noisy_props, noise_type, snr_db, output_dir):
    """创建原始音频和噪声音频的对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Audio Analysis: {noise_type} noise @ {snr_db}dB SNR', fontsize=16)
    
    # 时域波形对比
    axes[0, 0].set_title('Time Domain Comparison')
    if original_props:
        time_orig = np.linspace(0, original_props['duration'], original_props['length'])
        # 只显示前几秒避免图表过于密集
        display_samples = min(int(2 * original_props['sample_rate']), original_props['length'])
        axes[0, 0].plot(time_orig[:display_samples], 
                       np.linspace(-original_props['peak'], original_props['peak'], display_samples), 
                       'b-', alpha=0.7, label='Original (simulated)')
    
    if noisy_props:
        time_noisy = np.linspace(0, noisy_props['duration'], noisy_props['length'])
        display_samples = min(int(2 * noisy_props['sample_rate']), noisy_props['length'])
        axes[0, 0].plot(time_noisy[:display_samples], 
                       np.linspace(-noisy_props['peak'], noisy_props['peak'], display_samples), 
                       'r-', alpha=0.7, label='Noisy (simulated)')
    
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 频域对比
    axes[0, 1].set_title('Frequency Domain Comparison')
    if original_props:
        axes[0, 1].loglog(original_props['freqs'][1:], original_props['magnitude'][1:], 
                         'b-', alpha=0.7, label='Original')
    if noisy_props:
        axes[0, 1].loglog(noisy_props['freqs'][1:], noisy_props['magnitude'][1:], 
                         'r-', alpha=0.7, label='Noisy')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 统计信息对比
    axes[1, 0].set_title('Audio Statistics')
    categories = ['RMS', 'Peak', 'Duration']
    if original_props and noisy_props:
        orig_values = [original_props['rms'], original_props['peak'], original_props['duration']]
        noisy_values = [noisy_props['rms'], noisy_props['peak'], noisy_props['duration']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, orig_values, width, label='Original', alpha=0.7)
        axes[1, 0].bar(x + width/2, noisy_values, width, label='Noisy', alpha=0.7)
        
        axes[1, 0].set_xlabel('Properties')
        axes[1, 0].set_ylabel('Values')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(categories)
        axes[1, 0].legend()
    
    # SNR信息
    axes[1, 1].set_title('Noise Information')
    axes[1, 1].text(0.1, 0.8, f'Noise Type: {noise_type}', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.7, f'Target SNR: {snr_db} dB', transform=axes[1, 1].transAxes, fontsize=12)
    
    if original_props and noisy_props:
        # 计算实际SNR
        signal_power = original_props['rms'] ** 2
        noise_power = noisy_props['rms'] ** 2 - signal_power
        if noise_power > 0:
            actual_snr = 10 * np.log10(signal_power / noise_power)
            axes[1, 1].text(0.1, 0.6, f'Estimated Actual SNR: {actual_snr:.1f} dB', 
                           transform=axes[1, 1].transAxes, fontsize=12)
        
        axes[1, 1].text(0.1, 0.5, f'RMS Change: {(noisy_props["rms"]/original_props["rms"]-1)*100:.1f}%', 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.4, f'Peak Change: {(noisy_props["peak"]/original_props["peak"]-1)*100:.1f}%', 
                       transform=axes[1, 1].transAxes, fontsize=12)
    
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(output_dir, f'noise_verification_{noise_type}_{snr_db}db.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Verify noise injection results")
    parser.add_argument("--original_audio", required=True, help="Path to original audio file")
    parser.add_argument("--noisy_audio", required=True, help="Path to noisy audio file")
    parser.add_argument("--noise_type", required=True, help="Type of noise used")
    parser.add_argument("--snr_db", type=float, required=True, help="SNR level in dB")
    parser.add_argument("--output_dir", default=".", help="Output directory for verification results")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Real Noise Injection Verification")
    print("=" * 60)
    print(f"Original audio: {args.original_audio}")
    print(f"Noisy audio: {args.noisy_audio}")
    print(f"Noise type: {args.noise_type}")
    print(f"Target SNR: {args.snr_db} dB")
    print()
    
    # 检查文件是否存在
    if not os.path.exists(args.original_audio):
        print(f"❌ Original audio file not found: {args.original_audio}")
        return 1
    
    if not os.path.exists(args.noisy_audio):
        print(f"❌ Noisy audio file not found: {args.noisy_audio}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 分析音频文件
    print("Analyzing original audio...")
    original_props = analyze_audio_properties(args.original_audio)
    
    print("Analyzing noisy audio...")
    noisy_props = analyze_audio_properties(args.noisy_audio)
    
    if not original_props or not noisy_props:
        print("❌ Failed to analyze audio files")
        return 1
    
    # 打印分析结果
    print("\n" + "=" * 40)
    print("ANALYSIS RESULTS")
    print("=" * 40)
    
    print("\nOriginal Audio:")
    print(f"  Duration: {original_props['duration']:.2f} seconds")
    print(f"  Sample Rate: {original_props['sample_rate']} Hz")
    print(f"  RMS: {original_props['rms']:.4f}")
    print(f"  Peak: {original_props['peak']:.4f}")
    
    print("\nNoisy Audio:")
    print(f"  Duration: {noisy_props['duration']:.2f} seconds")
    print(f"  Sample Rate: {noisy_props['sample_rate']} Hz")
    print(f"  RMS: {noisy_props['rms']:.4f}")
    print(f"  Peak: {noisy_props['peak']:.4f}")
    
    # 计算变化
    duration_change = abs(noisy_props['duration'] - original_props['duration'])
    rms_change = (noisy_props['rms'] / original_props['rms'] - 1) * 100
    peak_change = (noisy_props['peak'] / original_props['peak'] - 1) * 100
    
    print("\nChanges:")
    print(f"  Duration change: {duration_change:.3f} seconds")
    print(f"  RMS change: {rms_change:+.1f}%")
    print(f"  Peak change: {peak_change:+.1f}%")
    
    # 估算实际SNR
    signal_power = original_props['rms'] ** 2
    total_power = noisy_props['rms'] ** 2
    noise_power = total_power - signal_power
    
    if noise_power > 0:
        actual_snr = 10 * np.log10(signal_power / noise_power)
        print(f"  Estimated actual SNR: {actual_snr:.1f} dB (target: {args.snr_db} dB)")
        snr_error = abs(actual_snr - args.snr_db)
        print(f"  SNR error: {snr_error:.1f} dB")
    else:
        print("  Warning: Could not estimate SNR (noise power ≤ 0)")
    
    # 验证结果
    print("\n" + "=" * 40)
    print("VERIFICATION RESULTS")
    print("=" * 40)
    
    success = True
    
    # 检查基本属性
    if duration_change > 0.01:  # 允许1ms的误差
        print("❌ Duration mismatch (should be identical)")
        success = False
    else:
        print("✅ Duration preserved")
    
    if noisy_props['sample_rate'] != original_props['sample_rate']:
        print("❌ Sample rate mismatch")
        success = False
    else:
        print("✅ Sample rate preserved")
    
    # 检查噪声是否添加成功
    if rms_change < 1:  # RMS应该增加至少1%
        print("❌ RMS change too small - noise may not have been added properly")
        success = False
    else:
        print("✅ Noise successfully added (RMS increased)")
    
    # 检查SNR
    if noise_power > 0:
        if snr_error > 3:  # 允许3dB的误差
            print(f"⚠️  SNR error large ({snr_error:.1f} dB) - may indicate processing issues")
        else:
            print("✅ SNR within acceptable range")
    
    # 生成对比图
    print("\nGenerating comparison plots...")
    plot_comparison(original_props, noisy_props, args.noise_type, args.snr_db, args.output_dir)
    
    # 最终结果
    print("\n" + "=" * 40)
    if success:
        print("🎉 VERIFICATION PASSED")
        print("Real noise injection appears to be working correctly!")
    else:
        print("❌ VERIFICATION FAILED")
        print("Issues detected with noise injection process.")
    print("=" * 40)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())