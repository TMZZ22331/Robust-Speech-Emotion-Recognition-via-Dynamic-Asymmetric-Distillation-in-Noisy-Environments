import os
import argparse
from pathlib import Path
import tqdm
import re

# 定义EmoDB数据集的情感映射
EMOTION_MAP = {
    'A': 'angry',    # 愤怒
    'T': 'happy',    # 高兴
    'L': 'sad',      # 悲伤
    'N': 'neutral',  # 中性
}

# 定义需要处理的情感类别（保留四种情感）
TARGET_EMOTIONS = set(EMOTION_MAP.values())

def parse_emodb_filename(filename):
    """
    解析EmoDB文件名格式：03a01Fa.wav
    返回: (speaker_id, sentence_type, sentence_num, emotion, variant)
    """
    # 移除文件扩展名
    basename = filename.replace('.wav', '')
    
    # 使用正则表达式解析文件名
    pattern = r'(\d+)([ab])(\d+)([A-Z])([a-z])'
    match = re.match(pattern, basename)
    
    if not match:
        return None
    
    speaker_id, sentence_type, sentence_num, emotion_code, variant = match.groups()
    
    # 转换为标准格式
    speaker_id = f'emodb_spk_{speaker_id}'
    emotion = EMOTION_MAP.get(emotion_code, None)
    
    return speaker_id, sentence_type, sentence_num, emotion, variant

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", required=True, type=str, help="EmoDB数据集的根目录"
    )
    parser.add_argument(
        "--dest", required=True, type=str, help="保存输出文件的目标目录"
    )
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    root_path = Path(args.root)
    dest_path = Path(args.dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    # 定义输出文件路径
    tsv_path = dest_path / "train.tsv"
    lbl_path = dest_path / "train.lbl"
    spk_path = dest_path / "train.spk"

    print(f"正在扫描EmoDB数据集目录: {root_path}")
    print(f"目标情感: {', '.join(TARGET_EMOTIONS)}")
    print(f"输出文件将保存到: {dest_path}")

    paths = []
    labels = []
    speakers = []
    
    # 获取所有wav文件
    wav_files = list(root_path.glob("*.wav"))
    
    print(f"找到 {len(wav_files)} 个wav文件")
    
    # 使用tqdm创建进度条
    for wav_file in tqdm.tqdm(wav_files, desc="处理音频文件"):
        # 解析文件名
        parsed = parse_emodb_filename(wav_file.name)
        
        if parsed is None:
            print(f"警告: 无法解析文件名 {wav_file.name}")
            continue
        
        speaker_id, sentence_type, sentence_num, emotion, variant = parsed
        
        # 只处理目标情感
        if emotion in TARGET_EMOTIONS:
            relative_path = wav_file.name
            paths.append(relative_path)
            labels.append(emotion)
            speakers.append(speaker_id)

    # 写入tsv文件
    with open(tsv_path, "w", encoding="utf-8") as f_tsv:
        # 写入真实的根目录路径
        f_tsv.write(f"{root_path.as_posix()}\n")
        for path in paths:
            f_tsv.write(f"{path}\n")

    # 写入lbl文件
    with open(lbl_path, "w", encoding="utf-8") as f_lbl:
        for label in labels:
            f_lbl.write(f"{label}\n")

    # 写入spk文件
    with open(spk_path, "w", encoding="utf-8") as f_spk:
        for speaker in speakers:
            f_spk.write(f"{speaker}\n")

    print("\nManifest文件生成完毕!")
    print(f"总共找到 {len(paths)} 个音频文件。")
    print(f"  - Manifest: {tsv_path}")
    print(f"  - Labels:   {lbl_path}")
    print(f"  - Speakers: {spk_path}")
    
    # 统计信息
    print("\n情感分布:")
    emotion_counts = {}
    for label in labels:
        emotion_counts[label] = emotion_counts.get(label, 0) + 1
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count} 个文件")
    
    print("\n说话人分布:")
    speaker_counts = {}
    for speaker in speakers:
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    for speaker, count in speaker_counts.items():
        print(f"  {speaker}: {count} 个文件")

if __name__ == "__main__":
    main() 