#!/usr/bin/env python3
"""
IEMOCAP数据集情感类别分布分析脚本
分析四种情感类别(ang, hap, neu, sad)在全集中的分布情况

🎯 功能:
   - 统计全集情感类别分布
   - 按Session分析分布情况
   - 生成可视化图表
   - 输出详细统计报告
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter, defaultdict
import json
from datetime import datetime

# 导入配置和数据加载模块
import config as cfg
from dataload_clean import load_ssl_features

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class IEMOCAPDistributionAnalyzer:
    """IEMOCAP数据集分布分析器"""
    
    def __init__(self):
        """初始化分析器"""
        print("🔍 初始化IEMOCAP数据集分布分析器...")
        self.data_path = cfg.CLEAN_DATA_DIR
        
        # 情感类别映射
        self.label_dict = cfg.LABEL_DICT
        self.class_names = cfg.CLASS_NAMES
        self.num_classes = cfg.NUM_CLASSES
        
        # 反向映射：数字标签到情感名称
        self.idx_to_emotion = {v: k for k, v in self.label_dict.items()}
        
        # 存储分析结果
        self.dataset_info = {}
        self.distribution_stats = {}
        self.session_stats = defaultdict(dict)
        
        print(f"📂 数据路径: {self.data_path}")
        print(f"🏷️  情感类别: {self.class_names}")
        print(f"🔢 类别映射: {self.label_dict}")
    
    def load_dataset(self):
        """加载完整的IEMOCAP数据集"""
        print("\n📊 加载IEMOCAP完整数据集...")
        
        try:
            # 使用dataload_clean.py中的load_ssl_features函数
            dataset = load_ssl_features(self.data_path, self.label_dict)
            
            self.feats = dataset['feats']
            self.sizes = dataset['sizes']
            self.offsets = dataset['offsets']
            self.labels = np.array(dataset['labels'])
            self.session_ids = dataset['session_ids']
            self.total_samples = dataset['num']
            
            # 基本信息
            self.dataset_info = {
                'total_samples': self.total_samples,
                'feature_dim': self.feats.shape[1],
                'unique_sessions': np.unique(self.session_ids).tolist(),
                'num_sessions': len(np.unique(self.session_ids))
            }
            
            print(f"✅ 数据集加载完成!")
            print(f"   📊 总样本数: {self.total_samples}")
            print(f"   📏 特征维度: {self.dataset_info['feature_dim']}")
            print(f"   🎭 Session数量: {self.dataset_info['num_sessions']}")
            print(f"   📋 Session列表: {self.dataset_info['unique_sessions']}")
            
        except Exception as e:
            print(f"❌ 数据集加载失败: {e}")
            raise
    
    def analyze_overall_distribution(self):
        """分析全集的情感类别分布"""
        print("\n📈 分析全集情感类别分布...")
        
        # 统计每个类别的样本数
        label_counts = Counter(self.labels)
        
        # 按照预定义的顺序整理
        ordered_counts = {}
        percentages = {}
        
        for emotion in self.class_names:
            emotion_idx = self.label_dict[emotion]
            count = label_counts[emotion_idx]
            ordered_counts[emotion] = count
            percentages[emotion] = (count / self.total_samples) * 100
        
        self.distribution_stats['overall'] = {
            'counts': ordered_counts,
            'percentages': percentages,
            'total_samples': self.total_samples
        }
        
        print("📋 全集情感类别分布:")
        print("-" * 50)
        for emotion in self.class_names:
            count = ordered_counts[emotion]
            pct = percentages[emotion]
            print(f"   {emotion.upper()}: {count:4d} 样本 ({pct:5.1f}%)")
        
        print(f"   总计: {self.total_samples:4d} 样本 (100.0%)")
        print("-" * 50)
    
    def analyze_session_distribution(self):
        """分析每个Session的情感类别分布"""
        print("\n🎭 分析各Session情感类别分布...")
        
        for session_id in sorted(self.dataset_info['unique_sessions']):
            if session_id is None:
                continue
                
            # 获取该session的样本索引
            session_mask = self.session_ids == session_id
            session_labels = self.labels[session_mask]
            session_sample_count = len(session_labels)
            
            # 统计该session的类别分布
            session_label_counts = Counter(session_labels)
            
            session_counts = {}
            session_percentages = {}
            
            for emotion in self.class_names:
                emotion_idx = self.label_dict[emotion]
                count = session_label_counts[emotion_idx]
                session_counts[emotion] = count
                session_percentages[emotion] = (count / session_sample_count) * 100 if session_sample_count > 0 else 0
            
            self.session_stats[session_id] = {
                'counts': session_counts,
                'percentages': session_percentages,
                'total_samples': session_sample_count
            }
            
            print(f"📊 Session {session_id} ({session_sample_count} 样本):")
            for emotion in self.class_names:
                count = session_counts[emotion]
                pct = session_percentages[emotion]
                print(f"   {emotion}: {count:3d} ({pct:5.1f}%)")
    
    def create_visualizations(self):
        """创建可视化图表"""
        print("\n🎨 生成可视化图表...")
        
        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 创建图表目录
        plots_dir = "iemocap_distribution_analysis"
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. 全集分布饼图
        self._create_overall_pie_chart(plots_dir)
        
        # 2. 全集分布柱状图
        self._create_overall_bar_chart(plots_dir)
        
        # 3. Session对比图
        self._create_session_comparison_chart(plots_dir)
        
        # 4. Session堆叠图
        self._create_session_stacked_chart(plots_dir)
        
        # 5. 详细统计表
        self._create_detailed_table(plots_dir)
        
        print(f"✅ 可视化图表已保存至: {plots_dir}/")
    
    def _create_overall_pie_chart(self, plots_dir):
        """创建全集分布饼图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        counts = [self.distribution_stats['overall']['counts'][emotion] for emotion in self.class_names]
        colors = sns.color_palette("husl", len(self.class_names))
        
        wedges, texts, autotexts = ax.pie(
            counts, 
            labels=self.class_names,
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*self.total_samples)})',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 12}
        )
        
        ax.set_title('IEMOCAP数据集情感类别分布\n(全集统计)', fontsize=16, fontweight='bold', pad=20)
        
        # 添加图例
        ax.legend(wedges, [f'{emotion.upper()}: {count}样本' 
                          for emotion, count in self.distribution_stats['overall']['counts'].items()],
                 title="情感类别",
                 loc="center left",
                 bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'overall_distribution_pie.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_overall_bar_chart(self, plots_dir):
        """创建全集分布柱状图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 样本数柱状图
        counts = [self.distribution_stats['overall']['counts'][emotion] for emotion in self.class_names]
        colors = sns.color_palette("husl", len(self.class_names))
        
        bars1 = ax1.bar(self.class_names, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('IEMOCAP情感类别样本数分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('情感类别', fontsize=12)
        ax1.set_ylabel('样本数', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # 在柱子上添加数值标签
        for bar, count in zip(bars1, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 百分比柱状图
        percentages = [self.distribution_stats['overall']['percentages'][emotion] for emotion in self.class_names]
        bars2 = ax2.bar(self.class_names, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_title('IEMOCAP情感类别百分比分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('情感类别', fontsize=12)
        ax2.set_ylabel('百分比 (%)', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
        
        # 在柱子上添加百分比标签
        for bar, pct in zip(bars2, percentages):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'overall_distribution_bars.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_session_comparison_chart(self, plots_dir):
        """创建Session对比图"""
        # 准备数据
        sessions = sorted([s for s in self.dataset_info['unique_sessions'] if s is not None])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # 1. 样本数对比
        session_data = []
        for session in sessions:
            for emotion in self.class_names:
                count = self.session_stats[session]['counts'][emotion]
                session_data.append({
                    'Session': f'Ses{session:02d}', 
                    'Emotion': emotion.upper(), 
                    'Count': count
                })
        
        df = pd.DataFrame(session_data)
        
        # 样本数分组柱状图
        sns.barplot(data=df, x='Session', y='Count', hue='Emotion', ax=ax1)
        ax1.set_title('各Session情感类别样本数分布对比', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Session', fontsize=12)
        ax1.set_ylabel('样本数', fontsize=12)
        ax1.legend(title='情感类别', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. 百分比对比
        percentage_data = []
        for session in sessions:
            for emotion in self.class_names:
                pct = self.session_stats[session]['percentages'][emotion]
                percentage_data.append({
                    'Session': f'Ses{session:02d}', 
                    'Emotion': emotion.upper(), 
                    'Percentage': pct
                })
        
        df_pct = pd.DataFrame(percentage_data)
        
        sns.barplot(data=df_pct, x='Session', y='Percentage', hue='Emotion', ax=ax2)
        ax2.set_title('各Session情感类别百分比分布对比', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Session', fontsize=12)
        ax2.set_ylabel('百分比 (%)', fontsize=12)
        ax2.legend(title='情感类别', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'session_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_session_stacked_chart(self, plots_dir):
        """创建Session堆叠图"""
        sessions = sorted([s for s in self.dataset_info['unique_sessions'] if s is not None])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 准备数据
        session_labels = [f'Ses{s:02d}' for s in sessions]
        
        # 1. 样本数堆叠图
        bottom_counts = np.zeros(len(sessions))
        colors = sns.color_palette("husl", len(self.class_names))
        
        for i, emotion in enumerate(self.class_names):
            counts = [self.session_stats[session]['counts'][emotion] for session in sessions]
            ax1.bar(session_labels, counts, bottom=bottom_counts, 
                   label=emotion.upper(), color=colors[i], alpha=0.8)
            bottom_counts += counts
        
        ax1.set_title('各Session情感类别样本数堆叠分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Session', fontsize=12)
        ax1.set_ylabel('样本数', fontsize=12)
        ax1.legend(title='情感类别')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. 百分比堆叠图
        bottom_pcts = np.zeros(len(sessions))
        
        for i, emotion in enumerate(self.class_names):
            percentages = [self.session_stats[session]['percentages'][emotion] for session in sessions]
            ax2.bar(session_labels, percentages, bottom=bottom_pcts, 
                   label=emotion.upper(), color=colors[i], alpha=0.8)
            bottom_pcts += percentages
        
        ax2.set_title('各Session情感类别百分比堆叠分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Session', fontsize=12)
        ax2.set_ylabel('百分比 (%)', fontsize=12)
        ax2.legend(title='情感类别')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'session_stacked_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_detailed_table(self, plots_dir):
        """创建详细统计表"""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # 准备表格数据
        sessions = sorted([s for s in self.dataset_info['unique_sessions'] if s is not None])
        
        # 表头
        headers = ['Session'] + [f'{emotion.upper()}\n(样本数|百分比)' for emotion in self.class_names] + ['总计']
        
        # 表格数据
        table_data = []
        
        # Session行
        for session in sessions:
            row = [f'Ses{session:02d}']
            for emotion in self.class_names:
                count = self.session_stats[session]['counts'][emotion]
                pct = self.session_stats[session]['percentages'][emotion]
                row.append(f'{count}\n({pct:.1f}%)')
            row.append(str(self.session_stats[session]['total_samples']))
            table_data.append(row)
        
        # 总计行
        total_row = ['总计']
        for emotion in self.class_names:
            count = self.distribution_stats['overall']['counts'][emotion]
            pct = self.distribution_stats['overall']['percentages'][emotion]
            total_row.append(f'{count}\n({pct:.1f}%)')
        total_row.append(str(self.total_samples))
        table_data.append(total_row)
        
        # 创建表格
        table = ax.table(cellText=table_data,
                        colLabels=headers,
                        cellLoc='center',
                        loc='center',
                        colColours=['lightgray'] * len(headers))
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # 设置最后一行（总计行）的样式
        for i in range(len(headers)):
            table[(len(table_data), i)].set_facecolor('lightblue')
            table[(len(table_data), i)].set_text_props(weight='bold')
        
        ax.set_title('IEMOCAP数据集详细分布统计表', fontsize=16, fontweight='bold', pad=20)
        
        plt.savefig(os.path.join(plots_dir, 'detailed_statistics_table.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_analysis_report(self):
        """保存分析报告"""
        print("\n💾 保存分析报告...")
        
        report_dir = "iemocap_distribution_analysis"
        os.makedirs(report_dir, exist_ok=True)
        
        # 准备报告数据
        report_data = {
            'analysis_info': {
                'dataset': 'IEMOCAP',
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_path': self.data_path,
                'analyzer_version': '1.0'
            },
            'dataset_info': self.dataset_info,
            'overall_distribution': self.distribution_stats['overall'],
            'session_distributions': dict(self.session_stats)
        }
        
        # 保存JSON报告
        report_path = os.path.join(report_dir, 'iemocap_distribution_analysis_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
        
        # 保存文本报告
        text_report_path = os.path.join(report_dir, 'iemocap_distribution_summary.txt')
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write("IEMOCAP数据集情感类别分布分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据路径: {self.data_path}\n")
            f.write(f"总样本数: {self.total_samples}\n")
            f.write(f"特征维度: {self.dataset_info['feature_dim']}\n")
            f.write(f"Session数量: {self.dataset_info['num_sessions']}\n\n")
            
            f.write("全集情感类别分布:\n")
            f.write("-" * 30 + "\n")
            for emotion in self.class_names:
                count = self.distribution_stats['overall']['counts'][emotion]
                pct = self.distribution_stats['overall']['percentages'][emotion]
                f.write(f"{emotion.upper()}: {count:4d} 样本 ({pct:5.1f}%)\n")
            f.write("-" * 30 + "\n\n")
            
            f.write("各Session分布:\n")
            f.write("-" * 30 + "\n")
            sessions = sorted([s for s in self.dataset_info['unique_sessions'] if s is not None])
            for session in sessions:
                f.write(f"Session {session} ({self.session_stats[session]['total_samples']} 样本):\n")
                for emotion in self.class_names:
                    count = self.session_stats[session]['counts'][emotion]
                    pct = self.session_stats[session]['percentages'][emotion]
                    f.write(f"  {emotion}: {count:3d} ({pct:5.1f}%)\n")
                f.write("\n")
        
        print(f"✅ 分析报告已保存:")
        print(f"   📊 JSON报告: {report_path}")
        print(f"   📝 文本报告: {text_report_path}")
    
    def run_complete_analysis(self):
        """运行完整的分析流程"""
        print("🚀 开始IEMOCAP数据集情感类别分布分析")
        print("=" * 60)
        
        try:
            # 1. 加载数据集
            self.load_dataset()
            
            # 2. 分析全集分布
            self.analyze_overall_distribution()
            
            # 3. 分析Session分布
            self.analyze_session_distribution()
            
            # 4. 创建可视化
            self.create_visualizations()
            
            # 5. 保存报告
            self.save_analysis_report()
            
            print("\n" + "=" * 60)
            print("🎉 IEMOCAP数据集分布分析完成!")
            print("📁 结果文件保存在 'iemocap_distribution_analysis/' 目录中")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ 分析过程中出现错误: {e}")
            raise


def main():
    """主函数"""
    print("IEMOCAP数据集情感类别分布分析工具")
    print("作者: AI Assistant")
    print("版本: 1.0")
    print("-" * 40)
    
    # 创建分析器并运行
    analyzer = IEMOCAPDistributionAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main() 