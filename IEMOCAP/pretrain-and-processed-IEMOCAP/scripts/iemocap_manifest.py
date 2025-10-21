#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
import os

import soundfile


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", metavar="DIR", 
        default='/path/to/IEMOCAP_full_release',
        help="root directory containing audio files to index"
    )
    parser.add_argument(
        "--dest", default="/path/to/manifest", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--label_path", default="/path/to/train.emo",
    )
    return parser


def main(args):
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)
    
    # 规范化路径，避免编码问题
    args.root = os.path.normpath(args.root)
    args.dest = os.path.normpath(args.dest) 
    args.label_path = os.path.normpath(args.label_path)
    
    root = os.path.join(args.root, 'Session{}')
    
    # 使用UTF-8编码打开文件，避免编码问题
    with open(args.label_path, 'r', encoding='utf-8') as rf, \
         open(os.path.join(args.dest, 'train.tsv'), 'w', encoding='utf-8') as wf:
        
        print(args.root, file=wf)
        for line in rf.readlines():
            line = line.strip()
            if not line:
                continue
                
            fname = line.split('\t')[0].strip()
            session = fname[4]  # 从Ses01开始提取数字
            folder = fname.rsplit('_', 1)[0]
            
            # 构建完整的音频文件路径
            fname_full = os.path.join(root.format(session), 'sentences', 'wav', folder, fname + '.wav')
            fname_full = os.path.normpath(fname_full)
            
            try:
                frames = soundfile.info(fname_full).frames
                # 计算相对路径
                suffix = os.path.relpath(fname_full, args.root)
                # 统一使用正斜杠（适用于tsv格式）
                suffix = suffix.replace('\\', '/')
                print(suffix, frames, sep='\t', file=wf)
            except Exception as e:
                print(f"Warning: Could not process file {fname_full}: {e}")
                continue

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
