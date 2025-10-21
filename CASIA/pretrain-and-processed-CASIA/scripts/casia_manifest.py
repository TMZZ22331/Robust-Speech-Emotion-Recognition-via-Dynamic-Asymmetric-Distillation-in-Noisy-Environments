import os
import argparse
from pathlib import Path
import tqdm

# 定义说话人ID的映射规则，以处理不一致的命名
SPEAKER_MAP = {
    # 相同文本300 -> 统一ID
    'liuchanhg': 'casia_spk_1',
    'wangzhe': 'casia_spk_2',
    'zhaoquanyin': 'casia_spk_3',
    'ZhaoZuoxiang': 'casia_spk_4',
    # 不同文本100 -> 统一ID
    'Chang.Liu': 'casia_spk_1',
    'Zhe.Wang': 'casia_spk_2',
    'Quanyin.Zhao': 'casia_spk_3',
    'Zuoxiang.Zhao': 'casia_spk_4',
}

# 定义需要处理的情感类别
TARGET_EMOTIONS = {'angry', 'happy', 'sad', 'neutral'}

# 定义情感标签映射规则，处理不同子文件夹中的标签命名不一致问题
EMOTION_MAP = {
    'angry': 'angry',
    'happy': 'happy', 
    'sad': 'sad',
    'neutral': 'neutral',
    'normal': 'neutral',  # 将"normal"映射为"neutral"
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", required=True, type=str, help="CASIA数据集的根目录"
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

    print(f"正在扫描CASIA数据集目录: {root_path}")
    print(f"目标情感: {', '.join(TARGET_EMOTIONS)}")
    print(f"情感标签映射: {EMOTION_MAP}")
    print(f"输出文件将保存到: {dest_path}")

    paths = []
    labels = []
    speakers = []
    
    # 遍历'相同文本300'和'不同文本100'两个子目录
    sub_dirs = ["相同文本300", "不同文本100"]
    
    # 使用tqdm创建进度条
    with tqdm.tqdm(total=len(sub_dirs), desc="扫描主目录") as pbar_main:
        for sub_dir in sub_dirs:
            pbar_main.set_description(f"扫描: {sub_dir}")
            current_path = root_path / sub_dir
            if not current_path.is_dir():
                print(f"警告: 找不到目录 {current_path}, 跳过...")
                pbar_main.update(1)
                continue

            speaker_folders = [f for f in current_path.iterdir() if f.is_dir()]
            for speaker_dir in tqdm.tqdm(speaker_folders, desc=f"  -> 扫描说话人", leave=False):
                speaker_id_original = speaker_dir.name
                
                # 标准化说话人ID
                if speaker_id_original not in SPEAKER_MAP:
                    continue # 如果说话人ID不在map中，则跳过
                
                speaker_id_normalized = SPEAKER_MAP[speaker_id_original]

                emotion_folders = [f for f in speaker_dir.iterdir() if f.is_dir()]
                for emotion_dir in emotion_folders:
                    emotion_original = emotion_dir.name.lower()
                    
                    # 使用映射规则处理情感标签
                    if emotion_original in EMOTION_MAP:
                        emotion_mapped = EMOTION_MAP[emotion_original]
                        for wav_file in emotion_dir.glob("*.wav"):
                            relative_path = wav_file.relative_to(root_path)
                            paths.append(str(relative_path).replace('\\', '/'))
                            labels.append(emotion_mapped)  # 使用映射后的标签
                            speakers.append(speaker_id_normalized)
            pbar_main.update(1)

    # 写入tsv文件
    with open(tsv_path, "w", encoding="utf-8") as f_tsv:
        # 写入真实的根目录路径，而不是"root"字符串
        f_tsv.write(f"{root_path.as_posix()}\n")
        for path in paths:
            # tsv文件只需要相对于--root的路径
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

if __name__ == "__main__":
    main() 