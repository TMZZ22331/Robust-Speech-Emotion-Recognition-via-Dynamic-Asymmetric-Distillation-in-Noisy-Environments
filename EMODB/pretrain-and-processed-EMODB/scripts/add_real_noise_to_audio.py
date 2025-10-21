#!/usr/bin/env python3
"""
为音频文件添加真实噪声 - EMODB版本
支持指定SNR (Signal-to-Noise Ratio) 的真实噪声添加
支持两种模式：
1. 按噪声类型分别处理 (type_specific)
2. 每个样本随机选择噪声类型 (random_noise)

基于IEMOCAP版本，但去除F16噪声类型
"""

import argparse
import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
import random
import glob
from pathlib import Path


def load_noise_files(noise_root):
    """
    加载所有噪声文件 - EMODB版本（去除F16噪声）
    仅支持指定目录下的四种标准噪声类型，找不到时直接报错
    
    Args:
        noise_root: 噪声文件根目录路径 (应该是 5types 目录)
    
    Returns:
        dict: 按噪声类型分组的噪声文件字典
    """
    # 强制使用指定的噪声文件夹
    expected_noise_root = r"C:\Users\admin\Desktop\DATA\noise-92\NOISE\data\signals\5types"
    if os.path.normpath(noise_root) != os.path.normpath(expected_noise_root):
        print(f"Warning: Using non-standard noise directory: {noise_root}")
        print(f"Expected: {expected_noise_root}")
    
    # 精确的文件名到噪声类型映射（去除F16噪声）
    NOISE_FILE_MAPPING = {
        "babble.wav": "babble",
        "factory1.wav": "factory",
        "hfchannel.wav": "hfchannel",
        "volvo.wav": "volvo"
    }
    
    print(f"Loading noise files from: {noise_root}")
    
    # 检查目录是否存在
    if not os.path.exists(noise_root):
        raise FileNotFoundError(f"[ERROR] 噪声目录不存在: {noise_root}")
    
    noise_files = {}
    missing_files = []
    
    # 检查每个必需的噪声文件
    for filename, noise_type in NOISE_FILE_MAPPING.items():
        file_path = os.path.join(noise_root, filename)
        
        if os.path.exists(file_path):
            if noise_type not in noise_files:
                noise_files[noise_type] = []
            noise_files[noise_type].append(file_path)
            print(f"  [OK] Found {noise_type}: {filename}")
        else:
            missing_files.append(filename)
            print(f"  [ERROR] Missing {noise_type}: {filename}")
    
    # 如果有缺失的噪声文件，直接报错
    if missing_files:
        raise FileNotFoundError(
            f"[ERROR] 缺失必需的噪声文件: {missing_files}\n"
            f"请确保以下文件存在于 {noise_root}:\n"
            f"  - babble.wav (人声嘈杂)\n" 
            f"  - factory1.wav (工厂噪声)\n"
            f"  - hfchannel.wav (高频信道)\n"
            f"  - volvo.wav (车辆噪声)"
        )
    
    # 检查是否所有四种噪声类型都找到了（去除F16后从5种改为4种）
    expected_types = set(NOISE_FILE_MAPPING.values())
    found_types = set(noise_files.keys())
    
    if len(found_types) != 4:
        raise ValueError(
            f"[ERROR] 噪声类型数量不匹配! 期望4种，实际找到{len(found_types)}种\n"
            f"期望类型: {expected_types}\n"
            f"找到类型: {found_types}"
        )
    
    print(f"[OK] 成功加载所有4种噪声类型（已去除F16），共 {sum(len(files) for files in noise_files.values())} 个文件")
    
    return noise_files


def load_noise_audio(noise_file, target_length):
    """
    加载噪声音频并调整到目标长度
    
    Args:
        noise_file: 噪声文件路径
        target_length: 目标长度（样本数）
    
    Returns:
        调整后的噪声音频
    """
    try:
        noise, sr = sf.read(noise_file)
        
        # 如果是立体声，转换为单声道
        if len(noise.shape) > 1:
            noise = np.mean(noise, axis=1)
        
        # 调整噪声长度
        if len(noise) < target_length:
            # 如果噪声太短，重复噪声
            repeat_times = int(np.ceil(target_length / len(noise)))
            noise = np.tile(noise, repeat_times)
        
        # 截取到目标长度
        noise = noise[:target_length]
        
        return noise
    except Exception as e:
        print(f"Error loading noise file {noise_file}: {e}")
        return None


def add_real_noise(audio, noise, snr_db):
    """
    给音频信号添加真实噪声
    
    Args:
        audio: 音频信号数组
        noise: 噪声信号数组
        snr_db: 信噪比(dB)，值越大噪声越小
    
    Returns:
        加噪后的音频信号
    """
    # 计算音频功率
    signal_power = np.mean(audio ** 2)
    
    # 计算噪声功率
    noise_power = np.mean(noise ** 2)
    
    # 根据SNR调整噪声强度
    snr_linear = 10 ** (snr_db / 10)
    target_noise_power = signal_power / snr_linear
    
    # 调整噪声功率
    if noise_power > 0:
        noise_scale = np.sqrt(target_noise_power / noise_power)
        scaled_noise = noise * noise_scale
    else:
        scaled_noise = noise
    
    # 添加噪声
    noisy_audio = audio + scaled_noise
    
    # 防止溢出，保持在[-1, 1]范围内
    max_val = np.max(np.abs(noisy_audio))
    if max_val > 1.0:
        noisy_audio = noisy_audio / max_val
    
    return noisy_audio


def process_single_audio(input_path, output_path, noise_files, snr_db, noise_mode="random", noise_type=None):
    """
    处理单个音频文件
    
    Args:
        input_path: 输入音频路径
        output_path: 输出音频路径
        noise_files: 噪声文件字典
        snr_db: 信噪比
        noise_mode: 噪声模式 ("random" 或 "type_specific")
        noise_type: 指定的噪声类型（仅在type_specific模式下使用）
    """
    try:
        # 读取音频
        audio, sr = sf.read(input_path)
        
        # 如果是立体声，转换为单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # 选择噪声文件 - 修复版本：移除fallback，严格匹配
        if noise_mode == "random":
            # 随机选择噪声类型和文件
            all_noise_files = []
            for files in noise_files.values():
                all_noise_files.extend(files)
            selected_noise_file = random.choice(all_noise_files)
            selected_noise_type = "random"
        else:  # type_specific
            # 使用指定的噪声类型 - 严格匹配，找不到就报错
            if noise_type not in noise_files:
                available_types = list(noise_files.keys())
                raise ValueError(
                    f"[ERROR] 噪声类型 '{noise_type}' 不存在!\n"
                    f"可用的噪声类型: {available_types}\n"
                    f"请检查噪声类型名称是否正确。\n"
                    f"支持的噪声类型: babble, factory, hfchannel, volvo（已去除F16）"
                )
            
            selected_noise_file = random.choice(noise_files[noise_type])
            selected_noise_type = noise_type
        
        # 打印选择的噪声信息（调试用）
        noise_filename = os.path.basename(selected_noise_file)
        if noise_mode == "type_specific":
            print(f"  Using {selected_noise_type} noise: {noise_filename}")
        else:
            print(f"  Using random noise: {noise_filename}")
        
        # 加载噪声
        noise = load_noise_audio(selected_noise_file, len(audio))
        if noise is None:
            print(f"Failed to load noise for {input_path}")
            return False
        
        # 添加噪声
        noisy_audio = add_real_noise(audio, noise, snr_db)
        
        # 保存加噪音频
        sf.write(output_path, noisy_audio, sr)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def get_parser():
    parser = argparse.ArgumentParser(description="Add real noise to audio files - EMODB version (no F16)")
    parser.add_argument(
        "--input_root", 
        required=True,
        help="Root directory of EMODB dataset"
    )
    parser.add_argument(
        "--output_root", 
        required=True,
        help="Output directory for noisy audio files"
    )
    parser.add_argument(
        "--noise_root",
        required=True,
        help="Root directory containing noise files"
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
    parser.add_argument(
        "--noise_mode",
        choices=["random", "type_specific"],
        default="random",
        help="Noise application mode: random (each sample gets random noise type) or type_specific (all samples get same noise type)"
    )
    parser.add_argument(
        "--noise_type",
        help="Specific noise type to use (only for type_specific mode). Available: babble, factory, hfchannel, volvo"
    )
    return parser


def main(args):
    print(f"Adding real noise to EMODB audio files (F16 噪声已去除)")
    print(f"Input root: {args.input_root}")
    print(f"Output root: {args.output_root}")
    print(f"Noise root: {args.noise_root}")
    print(f"SNR: {args.snr_db} dB")
    print(f"Noise mode: {args.noise_mode}")
    if args.noise_type:
        print(f"Noise type: {args.noise_type}")
    print(f"Manifest: {args.manifest_path}")
    
    # 创建输出目录结构
    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)
    
    # 加载噪声文件
    print("\nLoading noise files...")
    noise_files = load_noise_files(args.noise_root)
    if not noise_files:
        print("Error: No noise files found!")
        return 1
    
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
    
    # 设置随机种子保证可重现性
    np.random.seed(42)
    random.seed(42)
    
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
        if process_single_audio(input_path, output_path, noise_files, args.snr_db, 
                              args.noise_mode, args.noise_type):
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