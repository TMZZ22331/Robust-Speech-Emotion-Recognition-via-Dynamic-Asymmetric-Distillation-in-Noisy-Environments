import os
import argparse
from pathlib import Path
import shutil

def get_args():
    parser = argparse.ArgumentParser(
        description="为带噪声音频重新生成manifest文件。它会复用原始的.lbl和.spk文件"
    )
    parser.add_argument(
        "--root", required=True, type=str, 
        help="包含带噪声音频的临时根目录"
    )
    parser.add_argument(
        "--original-manifest-dir", required=True, type=str,
        help="包含原始train.tsv, train.lbl, train.spk文件的目录"
    )
    parser.add_argument(
        "--dest", required=True, type=str, 
        help="保存新manifest文件的目标目录"
    )
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    noisy_audio_root = Path(args.root)
    original_manifest_dir = Path(args.original_manifest_dir)
    dest_path = Path(args.dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    # 检查原始文件是否存在
    original_tsv = original_manifest_dir / "train.tsv"
    original_lbl = original_manifest_dir / "train.lbl"
    original_spk = original_manifest_dir / "train.spk"

    if not all([f.exists() for f in [original_tsv, original_lbl, original_spk]]):
        print(f"错误：在 {original_manifest_dir} 中找不到必需的原始manifest文件 (tsv, lbl, spk)。")
        exit(1)

    print(f"正在为带噪声音频生成新的manifest...")
    print(f"噪声音频根目录: {noisy_audio_root}")
    print(f"目标目录: {dest_path}")

    # 1. 读取原始的 tsv 文件 (跳过第一行列头 'root')
    with open(original_tsv, "r", encoding="utf-8") as f:
        # 保留完整的相对路径，保持目录结构
        original_paths = [line.strip() for line in f.readlines()[1:]]

    # 2. 一次性创建新的 tsv 文件
    new_tsv_path = dest_path / "train.tsv"
    with open(new_tsv_path, "w", encoding="utf-8") as f:
        # 写入新的根目录（带噪声音频的目录）
        f.write(f"{noisy_audio_root.as_posix()}\n")
        # 写入只有文件名的相对路径
        for file_name in original_paths:
            f.write(f"{file_name}\n")
    
    # 3. 直接复制 .lbl 和 .spk 文件，增加检查避免SameFileError
    dest_lbl = dest_path / "train.lbl"
    if not dest_lbl.exists() or not dest_lbl.samefile(original_lbl):
        shutil.copy(original_lbl, dest_lbl)

    dest_spk = dest_path / "train.spk"
    if not dest_spk.exists() or not dest_spk.samefile(original_spk):
        shutil.copy(original_spk, dest_spk)

    print("\n带噪声音频的Manifest文件已成功生成!")
    print(f"  - Manifest: {new_tsv_path}")
    print(f"  - Labels:   {dest_path / 'train.lbl'}")
    print(f"  - Speakers: {dest_path / 'train.spk'}")

if __name__ == "__main__":
    main()
