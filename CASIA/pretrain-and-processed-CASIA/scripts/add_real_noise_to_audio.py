#!/usr/bin/env python3
"""
为音频文件添加真实噪声
支持使用真实噪声文件进行噪声注入，可指定SNR
"""

import argparse
import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
import random
import glob


def load_noise_files(noise_dir):
    """
    从指定目录加载所有噪声文件
    
    Args:
        noise_dir: 噪声文件目录路径
    
    Returns:
        dict: 噪声类型到文件列表的映射
    """
    noise_files = {}
    
    # 支持的音频格式
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
    
    # 如果是直接包含噪声文件的目录
    all_noise_files = []
    for ext in audio_extensions:
        all_noise_files.extend(glob.glob(os.path.join(noise_dir, ext)))
    
    if all_noise_files:
        # 如果找到噪声文件，按文件名（去掉扩展名）分类
        for noise_file in all_noise_files:
            filename = os.path.basename(noise_file)
            # 提取噪声类型（去掉文件扩展名）
            noise_type = os.path.splitext(filename)[0]
            
            if noise_type not in noise_files:
                noise_files[noise_type] = []
            noise_files[noise_type].append(noise_file)
    else:
        # 如果没有直接找到文件，检查子目录
        for item in os.listdir(noise_dir):
            item_path = os.path.join(noise_dir, item)
            if os.path.isdir(item_path):
                # 获取子目录中的所有音频文件
                subdir_files = []
                for ext in audio_extensions:
                    subdir_files.extend(glob.glob(os.path.join(item_path, ext)))
                    subdir_files.extend(glob.glob(os.path.join(item_path, '**', ext), recursive=True))
                
                if subdir_files:
                    noise_files[item] = subdir_files
    
    return noise_files


def add_real_noise(audio, noise_audio, snr_db, sr_audio, sr_noise):
    """
    给音频信号添加真实噪声
    
    Args:
        audio: 原始音频信号数组
        noise_audio: 噪声信号数组
        snr_db: 信噪比(dB)
        sr_audio: 原始音频采样率
        sr_noise: 噪声音频采样率
    
    Returns:
        加噪后的音频信号
    """
    # 如果采样率不同，需要重采样噪声
    if sr_audio != sr_noise:
        # 简单的重采样：调整噪声长度
        ratio = sr_audio / sr_noise
        new_length = int(len(noise_audio) * ratio)
        noise_audio = np.interp(np.linspace(0, len(noise_audio), new_length), 
                               np.arange(len(noise_audio)), noise_audio)
    
    # 调整噪声长度匹配音频长度
    audio_len = len(audio)
    noise_len = len(noise_audio)
    
    if noise_len >= audio_len:
        # 如果噪声较长，随机选择一段
        start_idx = random.randint(0, noise_len - audio_len)
        noise_segment = noise_audio[start_idx:start_idx + audio_len]
    else:
        # 如果噪声较短，重复噪声
        repeat_times = (audio_len // noise_len) + 1
        repeated_noise = np.tile(noise_audio, repeat_times)
        noise_segment = repeated_noise[:audio_len]
    
    # 计算音频功率
    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(noise_segment ** 2)
    
    # 根据SNR调整噪声强度
    snr_linear = 10 ** (snr_db / 10)
    target_noise_power = signal_power / snr_linear
    
    # 避免除零错误
    if noise_power > 0:
        noise_scaling = np.sqrt(target_noise_power / noise_power)
        scaled_noise = noise_segment * noise_scaling
    else:
        scaled_noise = noise_segment
    
    # 添加噪声
    noisy_audio = audio + scaled_noise
    
    # 防止溢出，保持在[-1, 1]范围内
    max_val = np.max(np.abs(noisy_audio))
    if max_val > 1.0:
        noisy_audio = noisy_audio / max_val
    
    return noisy_audio


def process_single_audio(input_path, output_path, noise_files, noise_type, snr_db):
    """处理单个音频文件"""
    try:
        # 读取音频
        audio, sr_audio = sf.read(input_path)
        
        # 随机选择一个该类型的噪声文件
        if noise_type in noise_files and noise_files[noise_type]:
            noise_file = random.choice(noise_files[noise_type])
            
            # 读取噪声文件
            noise_audio, sr_noise = sf.read(noise_file)
            
            # 添加噪声
            noisy_audio = add_real_noise(audio, noise_audio, snr_db, sr_audio, sr_noise)
            
            # 保存加噪音频
            sf.write(output_path, noisy_audio, sr_audio)
            return True
        else:
            print(f"Warning: No noise files found for type '{noise_type}'")
            return False
            
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def get_parser():
    parser = argparse.ArgumentParser(description="Add real noise to audio files")
    parser.add_argument(
        "--input_root", 
        required=True,
        help="Root directory of audio dataset"
    )
    parser.add_argument(
        "--output_root", 
        required=True,
        help="Output directory for noisy audio files"
    )
    parser.add_argument(
        "--noise_dir", 
        required=True,
        help="Directory containing noise files"
    )
    parser.add_argument(
        "--noise_type", 
        required=True,
        help="Type of noise to add (should match subdirectory or file prefix in noise_dir)"
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
    print(f"Adding real noise to audio files")
    print(f"Input root: {args.input_root}")
    print(f"Output root: {args.output_root}")
    print(f"Noise directory: {args.noise_dir}")
    print(f"Noise type: {args.noise_type}")
    print(f"SNR: {args.snr_db} dB")
    print(f"Manifest: {args.manifest_path}")
    
    # 加载噪声文件
    print("Loading noise files...")
    noise_files = load_noise_files(args.noise_dir)
    
    if not noise_files:
        print(f"Error: No noise files found in {args.noise_dir}")
        return 1
    
    print("Available noise types:")
    for noise_type, files in noise_files.items():
        print(f"  {noise_type}: {len(files)} files")
    
    if args.noise_type not in noise_files:
        print(f"Error: Noise type '{args.noise_type}' not found in noise directory")
        print(f"Available types: {list(noise_files.keys())}")
        return 1
    
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
            relative_path = parts[0]
            audio_files.append(relative_path)
    
    print(f"Found {len(audio_files)} audio files to process")
    print(f"Using {len(noise_files[args.noise_type])} noise files of type '{args.noise_type}'")
    
    # 设置随机种子保证可部分重现性
    random.seed(42)
    np.random.seed(42)
    
    # 处理每个音频文件
    processed_count = 0
    failed_count = 0
    
    for relative_path in tqdm(audio_files, desc=f"Adding {args.noise_type} noise"):
        # 输入文件路径
        input_path = os.path.join(args.input_root, relative_path)
        
        # 输出文件路径（保持相同的目录结构）
        output_path = os.path.join(args.output_root, relative_path)
        
        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 处理音频文件
        if process_single_audio(input_path, output_path, noise_files, args.noise_type, args.snr_db):
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