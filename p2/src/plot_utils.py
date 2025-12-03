# plot_utils.py
"""
绘图工具模块 - 用于绘制训练和验证损失曲线
支持将不同超参数配置的曲线绘制在同一张图上
"""

import os
import matplotlib.pyplot as plt
import json
import glob

def plot_single_experiment(train_losses, val_losses, train_steps, val_steps, output_dir, args, experiment_name=None):
    """
    绘制单个实验的训练和验证损失曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_steps: 训练步骤列表
        val_steps: 验证步骤列表
        output_dir: 输出目录
        args: 训练参数
        experiment_name: 实验名称（可选）
    """
    if experiment_name is None:
        experiment_name = f"lr_{args.learning_rate}_bs_{args.batch_size}_epochs_{args.num_epochs}_opt_{args.optimization_method}"
        if args.optimization_method == "lora":
            experiment_name += f"_rank{args.lora_rank}"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制训练损失图
    plt.figure(figsize=(10, 6))
    if train_losses and train_steps:
        plt.plot(train_steps, train_losses, 'b-', label='Training Loss', alpha=0.7, linewidth=1)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve\n({experiment_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存训练损失图
    train_plot_path = os.path.join(output_dir, f'train_loss_{experiment_name}.png')
    plt.savefig(train_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制验证损失图
    plt.figure(figsize=(10, 6))
    if val_losses and val_steps:
        plt.plot(val_steps, val_losses, 'r-', label='Validation Loss', marker='o', markersize=4, linewidth=2)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(f'Validation Loss Curve\n({experiment_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存验证损失图
    val_plot_path = os.path.join(output_dir, f'val_loss_{experiment_name}.png')
    plt.savefig(val_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training loss curve saved to {train_plot_path}")
    print(f"Validation loss curve saved to {val_plot_path}")
    
    # 保存损失数据为JSON文件，便于后续合并绘图
    loss_data = {
        'experiment_name': experiment_name,
        'train_losses': train_losses,
        'train_steps': train_steps,
        'val_losses': val_losses,
        'val_steps': val_steps,
        'args': vars(args) if hasattr(args, '__dict__') else args
    }
    
    json_path = os.path.join(output_dir, f'loss_data_{experiment_name}.json')
    with open(json_path, 'w') as f:
        json.dump(loss_data, f, indent=2)
    
    return loss_data

def plot_multiple_experiments(experiments_data, output_dir, title_suffix=""):
    """
    将多个实验的训练和验证损失曲线绘制在同一张图上
    
    Args:
        experiments_data: 实验数据列表，每个元素为包含以下键的字典：
            - 'experiment_name': 实验名称
            - 'train_losses': 训练损失列表
            - 'train_steps': 训练步骤列表
            - 'val_losses': 验证损失列表
            - 'val_steps': 验证步骤列表
            - 'args': 实验参数
        output_dir: 输出目录
        title_suffix: 图表标题后缀
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义颜色和线型
    colors = ['orange', 'yellow', 'red', 'purple', 'blue', 'brown', 'pink', 'gray']
    line_styles = ['-', '--', '-.', ':']
    
    # 绘制多个实验的训练损失图
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(experiments_data):
        color = colors[i % len(colors)]
        line_style = line_styles[(i // len(colors)) % len(line_styles)]
        
        if data['train_losses'] and data['train_steps']:
            label = data.get('experiment_name', f'Experiment {i+1}')
            plt.plot(data['train_steps'], data['train_losses'], 
                    color=color, linestyle=line_style, 
                    label=label, alpha=0.8, linewidth=1.5)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Comparison{title_suffix}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存训练损失对比图
    train_comparison_path = os.path.join(output_dir, 'train_loss_comparison.png')
    plt.savefig(train_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制多个实验的验证损失图
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(experiments_data):
        color = colors[i % len(colors)]
        line_style = line_styles[(i // len(colors)) % len(line_styles)]
        
        if data['val_losses'] and data['val_steps']:
            label = data.get('experiment_name', f'Experiment {i+1}')
            plt.plot(data['val_steps'], data['val_losses'], 
                    color=color, linestyle=line_style, 
                    label=label, alpha=0.8, linewidth=2, marker='o', markersize=4)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(f'Validation Loss Comparison{title_suffix}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存验证损失对比图
    val_comparison_path = os.path.join(output_dir, 'val_loss_comparison.png')
    plt.savefig(val_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training loss comparison saved to {train_comparison_path}")
    print(f"Validation loss comparison saved to {val_comparison_path}")
    
    # 保存所有实验数据
    all_data_path = os.path.join(output_dir, 'all_experiments_data.json')
    with open(all_data_path, 'w') as f:
        json.dump(experiments_data, f, indent=2)
    
    return train_comparison_path, val_comparison_path

def load_experiment_data(experiment_dir):
    """
    从实验目录加载损失数据
    
    Args:
        experiment_dir: 实验目录路径
        
    Returns:
        加载的实验数据字典
    """
    json_files = glob.glob(os.path.join(experiment_dir, 'loss_data_*.json'))
    
    if not json_files:
        print(f"No loss data found in {experiment_dir}")
        return None
    
    # 加载最新的数据文件
    latest_file = max(json_files, key=os.path.getctime)
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    return data

def compare_experiments(experiment_dirs, output_dir="comparison_results", title_suffix=""):
    """
    比较多个实验的损失曲线
    
    Args:
        experiment_dirs: 实验目录列表
        output_dir: 输出目录
        title_suffix: 图表标题后缀
    """
    experiments_data = []
    
    for exp_dir in experiment_dirs:
        data = load_experiment_data(exp_dir)
        if data:
            experiments_data.append(data)
    
    if not experiments_data:
        print("No experiment data loaded. Please check the directories.")
        return
    
    print(f"Loaded data from {len(experiments_data)} experiments")
    
    # 绘制对比图
    train_path, val_path = plot_multiple_experiments(
        experiments_data, output_dir, title_suffix
    )
    
    return train_path, val_path