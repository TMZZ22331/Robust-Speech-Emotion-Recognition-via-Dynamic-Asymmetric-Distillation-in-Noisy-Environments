#!/usr/bin/env python3
"""
为音频文件添加白噪声
支持指定SNR (Signal-to-Noise Ratio) 的白噪声添加
"""

import argparse
import os
import numpy as np
import soundfile as sf
from tqdm import tqdm


def add_white_noise(audio, snr_db):
    """
    给音频信号添加白噪声
    
    Args:
        audio: 音频信号数组
        snr_db: 信噪比(dB)，值越大噪声越小
    
    Returns:
        加噪后的音频信号
    """
    # 计算音频功率
    signal_power = np.mean(audio ** 2)
    
    # 根据SNR计算噪声功率
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # 生成白噪声
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    
    # 添加噪声
    noisy_audio = audio + noise
    
    # 防止溢出，保持在[-1, 1]范围内
    max_val = np.max(np.abs(noisy_audio))
    if max_val > 1.0:
        noisy_audio = noisy_audio / max_val
    
    return noisy_audio


def process_single_audio(input_path, output_path, snr_db):
    """处理单个音频文件"""
    try:
        # 读取音频
        audio, sr = sf.read(input_path)
        
        # 添加噪声
        noisy_audio = add_white_noise(audio, snr_db)
        
        # 保存加噪音频
        sf.write(output_path, noisy_audio, sr)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def get_parser():
    parser = argparse.ArgumentParser(description="Add white noise to audio files")
    parser.add_argument(
        "--input_root", 
        required=True,
        help="Root directory of IEMOCAP dataset"
    )
    parser.add_argument(
        "--output_root", 
        required=True,
        help="Output directory for noisy audio files"
    )
    parser.add_argument(
        "--snr_db", 
        type=float, 
        default=20.0,
        help="Signal-to-Noise Ratio in dB (default: 20.0)"
    )
    parser.add_argument(
        "--manifest_path",
        required=True,
        help="Path to train.tsv manifest file"
    )
    return parser


def main(args):
    print(f"Adding white noise to IEMOCAP audio files")
    print(f"Input root: {args.input_root}")
    print(f"Output root: {args.output_root}")
    print(f"SNR: {args.snr_db} dB")
    print(f"Manifest: {args.manifest_path}")
    
    # 创建输出目录结构
    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)
    
    # 读取manifest文件获取音频文件列表
    if not os.path.exists(args.manifest_path):
        print(f"Error: Manifest file not found: {args.manifest_path}")
        return 1
    
    audio_files = []
    with open(args.manifest_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        root_path = lines[0].strip()  # 第一行是root路径
        
        for line in lines[1:]:  # 跳过第一行
            parts = line.strip().split('\t')
            # 兼容没有制表符分割的情况
            relative_path = parts[0]
            audio_files.append(relative_path)
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # 设置随机种子保证可重现性
    np.random.seed(42)
    
    # 处理每个音频文件
    processed_count = 0
    failed_count = 0
    
    for relative_path in tqdm(audio_files, desc="Adding noise"):
        # 输入文件路径
        input_path = os.path.join(args.input_root, relative_path)
        
        # 输出文件路径（保持相同的目录结构）
        output_path = os.path.join(args.output_root, relative_path)
        
        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 处理音频文件
        if process_single_audio(input_path, output_path, args.snr_db):
            processed_count += 1
        else:
            failed_count += 1
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {processed_count} files")
    print(f"Failed: {failed_count} files")
    
    if processed_count > 0:
        print(f"Noisy audio files saved to: {args.output_root}")
        return 0
    else:
        print("No files were processed successfully!")
        return 1


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    exit(main(args)) 