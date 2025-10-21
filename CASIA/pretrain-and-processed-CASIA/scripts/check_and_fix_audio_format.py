#!/usr/bin/env python3
"""
检查和修复音频格式问题
确保音频符合emotion2vec特征提取的要求：
- 采样率: 16kHz
- 声道: 单声道
- 格式: WAV
"""

import argparse
import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
import glob
from pathlib import Path


def check_audio_format(file_path):
    """
    检查音频文件格式
    
    Returns:
        dict: 包含音频信息和是否需要修复的字典
    """
    try:
        info = sf.info(file_path)
        
        needs_fix = False
        issues = []
        
        if info.samplerate != 16000:
            needs_fix = True
            issues.append(f"采样率错误: {info.samplerate}Hz (应为16000Hz)")
        
        if info.channels != 1:
            needs_fix = True
            issues.append(f"声道数错误: {info.channels} (应为1)")
        
        if info.duration == 0:
            needs_fix = True
            issues.append("音频时长为0")
        
        return {
            'path': file_path,
            'samplerate': info.samplerate,
            'channels': info.channels,
            'duration': info.duration,
            'frames': info.frames,
            'needs_fix': needs_fix,
            'issues': issues
        }
        
    except Exception as e:
        return {
            'path': file_path,
            'samplerate': None,
            'channels': None,
            'duration': None,
            'frames': None,
            'needs_fix': True,
            'issues': [f"无法读取文件: {str(e)}"]
        }


def fix_audio_format(input_path, output_path=None):
    """
    修复音频格式
    
    Args:
        input_path: 输入音频路径
        output_path: 输出音频路径（如果为None，则覆盖原文件）
    
    Returns:
        bool: 是否修复成功
    """
    try:
        # 读取音频
        audio, sr = sf.read(input_path)
        
        # 转换为单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # 重采样到16kHz（简单的线性插值）
        if sr != 16000:
            # 使用scipy.signal.resample会更好，但为了避免依赖，这里使用简单方法
            target_length = int(len(audio) * 16000 / sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, target_length),
                np.arange(len(audio)),
                audio
            )
            sr = 16000
        
        # 确保音频不为空
        if len(audio) == 0:
            print(f"警告: 音频文件为空 {input_path}")
            return False
        
        # 归一化音频（防止溢出）
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95
        
        # 保存音频
        if output_path is None:
            output_path = input_path
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        sf.write(output_path, audio, sr)
        return True
        
    except Exception as e:
        print(f"修复音频失败 {input_path}: {str(e)}")
        return False


def scan_manifest_files(manifest_path):
    """
    从manifest文件中获取音频文件列表
    
    Args:
        manifest_path: manifest文件路径（train.tsv）
    
    Returns:
        list: 音频文件路径列表
    """
    if not os.path.exists(manifest_path):
        print(f"Manifest文件不存在: {manifest_path}")
        return []
    
    audio_files = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) < 2:
            print(f"Manifest文件格式错误: {manifest_path}")
            return []
        
        root_path = lines[0].strip()
        for line in lines[1:]:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) > 0:
                    audio_path = os.path.join(root_path, parts[0])
                    audio_files.append(audio_path)
    
    return audio_files


def get_parser():
    parser = argparse.ArgumentParser(description="检查和修复音频格式问题")
    parser.add_argument(
        "--mode",
        choices=["check", "fix", "manifest"],
        required=True,
        help="操作模式: check(仅检查), fix(检查并修复), manifest(基于manifest文件检查)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="输入路径（目录路径或manifest文件路径）"
    )
    parser.add_argument(
        "--output",
        help="输出目录（可选，如果不指定则覆盖原文件）"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归搜索子目录中的音频文件"
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    print(f"音频格式检查工具")
    print(f"模式: {args.mode}")
    print(f"输入: {args.input}")
    if args.output:
        print(f"输出: {args.output}")
    print("")
    
    # 获取音频文件列表
    audio_files = []
    
    if args.mode == "manifest":
        # 从manifest文件获取
        audio_files = scan_manifest_files(args.input)
    else:
        # 从目录获取
        if os.path.isfile(args.input) and args.input.endswith('.wav'):
            audio_files = [args.input]
        elif os.path.isdir(args.input):
            pattern = "**/*.wav" if args.recursive else "*.wav"
            audio_files = glob.glob(os.path.join(args.input, pattern), recursive=args.recursive)
        else:
            print(f"无效的输入路径: {args.input}")
            return 1
    
    if not audio_files:
        print("未找到音频文件")
        return 1
    
    print(f"找到 {len(audio_files)} 个音频文件")
    print("")
    
    # 检查音频文件
    issues_count = 0
    fixed_count = 0
    failed_count = 0
    
    for audio_file in tqdm(audio_files, desc="处理音频文件"):
        info = check_audio_format(audio_file)
        
        if info['needs_fix']:
            issues_count += 1
            print(f"\n问题文件: {audio_file}")
            for issue in info['issues']:
                print(f"  - {issue}")
            
            if args.mode == "fix":
                # 确定输出路径
                if args.output:
                    rel_path = os.path.relpath(audio_file, args.input)
                    output_path = os.path.join(args.output, rel_path)
                else:
                    output_path = audio_file
                
                # 修复音频
                if fix_audio_format(audio_file, output_path):
                    print(f"  ✅ 已修复")
                    fixed_count += 1
                else:
                    print(f"  ❌ 修复失败")
                    failed_count += 1
    
    # 打印总结
    print(f"\n" + "="*50)
    print(f"检查完成！")
    print(f"总文件数: {len(audio_files)}")
    print(f"有问题的文件: {issues_count}")
    
    if args.mode == "fix":
        print(f"成功修复: {fixed_count}")
        print(f"修复失败: {failed_count}")
        print(f"无需修复: {len(audio_files) - issues_count}")
    
    if issues_count == 0:
        print("🎉 所有音频文件格式正确！")
        return 0
    elif args.mode == "check":
        print("⚠️ 发现音频格式问题，请使用 --mode fix 进行修复")
        return 1
    elif fixed_count == issues_count:
        print("🎉 所有问题已修复！")
        return 0
    else:
        print("⚠️ 部分文件修复失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    exit(main())