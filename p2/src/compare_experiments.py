# compare_experiments.py
"""
实验比较脚本 - 用于比较不同超参数配置的训练结果
"""

import os
import argparse
from plot_utils import compare_experiments

def main():
    parser = argparse.ArgumentParser(description="Compare multiple experiment results")
    
    parser.add_argument('--experiment_dirs', nargs='+', required=True,
                       help='List of experiment directories to compare')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                       help='Directory to save comparison plots')
    parser.add_argument('--title_suffix', type=str, default='',
                       help='Suffix to add to plot titles')
    
    args = parser.parse_args()
    
    # 确保实验目录存在
    valid_dirs = []
    for exp_dir in args.experiment_dirs:
        if os.path.exists(exp_dir):
            valid_dirs.append(exp_dir)
        else:
            print(f"Warning: Experiment directory {exp_dir} does not exist")
    
    if not valid_dirs:
        print("Error: No valid experiment directories provided")
        return
    
    print(f"Comparing {len(valid_dirs)} experiments: {valid_dirs}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 比较实验
    train_path, val_path = compare_experiments(
        valid_dirs, args.output_dir, args.title_suffix
    )
    
    print(f"\nComparison complete!")
    print(f"Training comparison saved to: {train_path}")
    print(f"Validation comparison saved to: {val_path}")

if __name__ == '__main__':
    main()