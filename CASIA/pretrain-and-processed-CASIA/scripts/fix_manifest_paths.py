#!/usr/bin/env python3
"""
修复manifest文件中的路径引用
将错误的平铺路径引用修复为正确的层次化路径引用
"""

import os
import glob
from pathlib import Path


def fix_manifest_paths(data_dir):
    """
    修复manifest文件中的路径引用
    
    Args:
        data_dir: 数据目录路径（例如：processed_features_CASIA_noisy/root1-factory-20db）
    """
    manifest_path = os.path.join(data_dir, "train.tsv")
    labels_path = os.path.join(data_dir, "train.lbl")
    speakers_path = os.path.join(data_dir, "train.spk")
    
    if not os.path.exists(manifest_path):
        print(f"错误：找不到manifest文件 {manifest_path}")
        return False
    
    # 获取所有实际存在的音频文件
    noisy_audio_temp = os.path.join(data_dir, "noisy_audio_temp")
    if not os.path.exists(noisy_audio_temp):
        print(f"错误：找不到噪声音频目录 {noisy_audio_temp}")
        return False
    
    # 递归查找所有wav文件
    actual_files = []
    for root, dirs, files in os.walk(noisy_audio_temp):
        for file in files:
            if file.endswith('.wav'):
                full_path = os.path.join(root, file)
                # 获取相对于noisy_audio_temp的路径
                rel_path = os.path.relpath(full_path, noisy_audio_temp)
                actual_files.append(rel_path.replace('\\', '/'))  # 统一使用正斜杠
    
    actual_files.sort()
    print(f"找到 {len(actual_files)} 个实际音频文件")
    
    # 读取现有的标签和说话人信息
    labels = []
    speakers = []
    
    if os.path.exists(labels_path):
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines()]
    
    if os.path.exists(speakers_path):
        with open(speakers_path, 'r', encoding='utf-8') as f:
            speakers = [line.strip() for line in f.readlines()]
    
    # 如果标签和说话人信息数量与音频文件数量匹配，则重新生成manifest
    if len(labels) == len(actual_files) and len(speakers) == len(actual_files):
        print("标签和说话人信息数量匹配，重新生成manifest文件...")
    else:
        print(f"警告：标签数量({len(labels)})、说话人数量({len(speakers)})与音频文件数量({len(actual_files)})不匹配")
        print("将基于文件路径重新生成标签和说话人信息...")
        
        # 从文件路径中提取标签和说话人信息
        labels = []
        speakers = []
        
        for file_path in actual_files:
            # 文件路径格式：不同文本100/Chang.Liu/angry/001.wav
            parts = file_path.split('/')
            if len(parts) >= 3:
                speaker = parts[1]  # Chang.Liu
                emotion = parts[2]  # angry
            else:
                speaker = "unknown"
                emotion = "neutral"
            
            speakers.append(speaker)
            labels.append(emotion)
    
    # 生成新的manifest文件
    noisy_audio_temp_abs = os.path.abspath(noisy_audio_temp).replace('\\', '/')
    
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write(noisy_audio_temp_abs + '\n')
        for file_path in actual_files:
            f.write(file_path + '\n')
    
    # 写入标签文件
    with open(labels_path, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(label + '\n')
    
    # 写入说话人文件
    with open(speakers_path, 'w', encoding='utf-8') as f:
        for speaker in speakers:
            f.write(speaker + '\n')
    
    print(f"✅ 已修复manifest文件！")
    print(f"  - Manifest: {manifest_path}")
    print(f"  - Labels:   {labels_path}")
    print(f"  - Speakers: {speakers_path}")
    print(f"  - 文件数量: {len(actual_files)}")
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="修复manifest文件中的路径引用")
    parser.add_argument("--data_dir", required=True, help="数据目录路径")
    args = parser.parse_args()
    
    success = fix_manifest_paths(args.data_dir)
    if success:
        print("✅ 修复完成！")
    else:
        print("❌ 修复失败！")
        exit(1)