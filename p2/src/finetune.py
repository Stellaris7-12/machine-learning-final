import os
import pickle #  导入pickle模块，用于Python对象的序列化和反序列化
import argparse #  导入argparse模块，用于解析命令行参数
from tqdm import tqdm #  从tqdm库中导入tqdm，用于在循环中显示进度条
import torch
import torch.optim as optim #  导入PyTorch的优化器模块，包含了各种优化算法
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence #  导入pad_sequence函数，用于填充变长序列
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, #  导入AutoModelForCausalLM类，用于自动加载因果语言模型
    AutoTokenizer #  导入AutoTokenizer类，用于自动加载对应的分词器
)
from peft import LoraConfig, get_peft_model #  导入PEFT库中的LoraConfig和get_peft_model，用于模型参数高效微调

# 导入绘图工具
from plot_utils import plot_single_experiment
import time

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(225040182)

def get_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a language model.")

    # Model and Data paths
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-Math-1.5B', help='The name of the pretrained model to use.') # 原来是Qwen/Qwen3-0.6B-Base
    parser.add_argument('--data_dir', type=str, default='data', help='Directory where the data is stored.')
    parser.add_argument('--output_dir', type=str, default='out-instruction-tuning', help='Directory to save the fine-tuned model.')

    # Training Hyperparameters
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for the optimizer.')
    parser.add_argument('--beta1', type=float, default=0.9, help='AdamW optimizer beta1.')
    parser.add_argument('--beta2', type=float, default=0.95, help='AdamW optimizer beta2.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training and validation.')
    parser.add_argument('--grad_accumulation_steps', type=int, default=4, help='Number of steps to accumulate gradients.')

    # Logging and Evaluation
    parser.add_argument('--log_interval', type=int, default=10, help='Log training loss every N steps.')
    parser.add_argument('--eval_interval', type=int, default=50, help='Run validation every N steps.')
    
    # Logging verbosity
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2], 
                       help='Verbosity level: 0=silent, 1=minimal, 2=detailed progress bars.')

    # Optimization method
    parser.add_argument('--optimization_method', type=str, default='adam', choices=['adam', 'sgd', 'lora'], help='Optimization method to use.')
    parser.add_argument('--lora_rank', type=int, default=8, help='The rank of the LoRA matrices.')
    
    # Experiment name for plotting
    parser.add_argument('--experiment_name', type=str, default=None, help='Custom name for this experiment (for plotting).')

    return parser.parse_args()

class TokenizedDataset(Dataset):
    """A simple dataset class to load tokenized IDs from a pickle file."""
    def __init__(self, pickle_file_path):
        if not os.path.exists(pickle_file_path):
            raise FileNotFoundError(
                f"Pickle file not found at {pickle_file_path}. "
                "Please run the data preparation script first."
            )
        with open(pickle_file_path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Loaded {len(self.data)} examples from {pickle_file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SmartDataCollator:
    """
    Pads sequences to the max length in a batch and creates labels.
    Labels are -100 for pad tokens.
    """
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        attention_masks = [torch.tensor(item['attention_mask']) for item in batch]
        labels = [torch.tensor(item['labels']) for item in batch]

        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_masks,
            'labels': padded_labels
        }

class SimpleTqdmWrapper:
    """
    简化版的进度条包装器，根据verbose级别控制显示
    """
    def __init__(self, iterable, desc, verbose=1, total=None):
        self.iterable = iterable
        self.desc = desc
        self.verbose = verbose
        self.total = total
        self.current = 0
        
    def __iter__(self):
        if self.verbose >= 2:
            # 详细模式：显示完整进度条
            return iter(tqdm(self.iterable, desc=self.desc, total=self.total))
        elif self.verbose == 1:
            # 最小模式：显示简化进度条
            return iter(self.simple_tqdm())
        else:
            # 静默模式：不显示进度条
            return iter(self.iterable)
    
    def simple_tqdm(self):
        """简化版的进度条，只显示开始和结束"""
        print(f"{self.desc}...", end="", flush=True)
        for i, item in enumerate(self.iterable):
            yield item
        print(f" Done ({self.total} steps)")

def main():
    args = get_args()
    
    # 记录开始时间
    start_time = time.time()
    # 记录内存使用情况
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    # Derived paths
    current_file_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file_path)
    data_dir = os.path.join(script_dir, args.data_dir)
    train_data_path = os.path.join(data_dir, 'train.pkl')
    val_data_path = os.path.join(data_dir, 'val.pkl')
    output_dir = os.path.join(script_dir, args.output_dir)

    # 初始化损失记录
    train_losses = []      # 记录训练损失
    train_steps = []       # 记录训练损失对应的步骤
    val_losses = []        # 记录验证损失  
    val_steps = []         # 记录验证时的步骤数

    print(f"Loading model and tokenizer from {args.model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    # 使用预训练模型的分词器
    tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,  # 预训练模型的名称或路径
    trust_remote_code=True,  # 允许执行远程代码
    padding_side='left',  # 设置填充在文本左侧
    fix_mistral_regex=True  # 新增：用于修复Mistral模型正则表达式的配置参数
    )
    # 加载预训练的因果语言模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,  # 预训练模型的名称或路径
        trust_remote_code=True,  # 允许执行远程代码
        dtype=dtype  # 指定模型的数据类型
    ).to(device)  # 将模型移动到指定的设备（GPU/CPU）

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        print("Set pad_token to eos_token")

    collate_fn = SmartDataCollator(pad_token_id=tokenizer.pad_token_id)

    train_dataset = TokenizedDataset(train_data_path)
    val_dataset = TokenizedDataset(val_data_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    print(f"Setting up optimizer: {args.optimization_method}")

    # Apply different optimizer
    if args.optimization_method == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )
    elif args.optimization_method == "adamw":
        optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=1e-8  # 通常添加eps参数
        )
    elif args.optimization_method == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimization_method == "lora": # AdamW + lora
        print(f"Setting up LoRA with rank={args.lora_rank}")
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            bias="none",
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", 'down_proj'], # Apply Lora to all possible modules
        )
        model = get_peft_model(model, lora_config)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )
    else:
        raise ValueError(f"Unknown optimization_method: {args.optimization_method}")

    print("Starting training...")
    best_val_loss = float('inf')
    global_step = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 计算总步数
    total_train_steps = len(train_loader) * args.num_epochs // args.grad_accumulation_steps
    print(f"Total training steps: {total_train_steps}")
    
    for epoch in range(args.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{args.num_epochs} ---")
        model.train()
        
        # 根据verbose级别选择进度条显示方式
        if args.verbose >= 2:
            # 详细模式：显示完整进度条
            train_iter = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", 
                             total=len(train_loader), leave=False)
        else:
            # 简化模式或静默模式
            train_iter = train_loader
            
        for step, batch in enumerate(train_iter):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=dtype):
                outputs = model(**batch)
                loss = outputs.loss
            loss = loss / args.grad_accumulation_steps
            loss.backward()
            if (step + 1) % args.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # 记录训练损失
                if global_step % args.log_interval == 0:
                    current_loss = loss.item() * args.grad_accumulation_steps
                    train_losses.append(current_loss)
                    train_steps.append(global_step)
                    
                    # 只在详细模式下显示每次损失
                    if args.verbose >= 2:
                        print(f"Step {global_step}: Train Loss = {current_loss:.4f}")
                    elif args.verbose == 1 and global_step % (args.log_interval * 5) == 0:
                        # 在最小模式下，每5次记录显示一次
                        print(f"Step {global_step}: Train Loss = {current_loss:.4f}")
                
                # 验证
                if global_step % args.eval_interval == 0:
                    model.eval()
                    
                    # 根据verbose级别选择验证进度条显示方式
                    if args.verbose >= 2:
                        val_iter = tqdm(val_loader, desc="Validating", 
                                       total=len(val_loader), leave=False)
                    else:
                        val_iter = val_loader
                    
                    total_val_loss = 0
                    with torch.no_grad():
                        for val_batch in val_iter:
                            val_batch = {k: v.to(device) for k, v in val_batch.items()}
                            with torch.autocast(device_type=device, dtype=dtype):
                                val_outputs = model(**val_batch)
                                val_loss = val_outputs.loss
                            total_val_loss += val_loss.item()
                    
                    avg_val_loss = total_val_loss / len(val_loader)
                    val_losses.append(avg_val_loss)
                    val_steps.append(global_step)
                    
                    print(f"\nStep {global_step}: Train Loss = {current_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
                    
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        print(f"  ✓ New best! Saving model to {output_dir}")
                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                    
                    model.train()

    print("\nTraining finished. Running one final evaluation...")
    model.eval()
    total_val_loss = 0
    
    # 最终验证的进度条
    if args.verbose >= 2:
        val_iter = tqdm(val_loader, desc="Final Validation", total=len(val_loader))
    else:
        print("Final validation...", end="", flush=True)
        val_iter = val_loader
    
    with torch.no_grad():
        for val_batch in val_iter:
            val_batch = {k: v.to(device) for k, v in val_batch.items()}
            with torch.autocast(device_type=device, dtype=dtype):
                val_outputs = model(**val_batch)
                val_loss = val_outputs.loss
            total_val_loss += val_loss.item()
    
    if args.verbose <= 1:
        print(" Done")
    
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"\nFinal Validation Loss = {avg_val_loss:.4f}")
    
    if avg_val_loss < best_val_loss:
        print(f"  ✓ Final model is the best! Saving to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        print(f"  ⓘ An earlier checkpoint was better (Val Loss: {best_val_loss:.4f})")

    # 绘制损失曲线并保存数据
    print("\nPlotting loss curves...")
    plot_single_experiment(
        train_losses, val_losses, train_steps, val_steps, 
        output_dir, args, args.experiment_name
    )

    # 计算训练时间和内存使用
    training_time = time.time() - start_time
    # 打印训练摘要
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Model: {args.model_name}")
    print(f"Optimizer: {args.optimization_method}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Final Training Steps: {global_step}")
    print(f"Loss data saved for {len(train_losses)} training points and {len(val_losses)} validation points")
    if torch.cuda.is_available():
        max_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Training Time: {training_time:.1f}s ({training_time/60:.1f}min)")
        print(f"Peak GPU memory: {max_memory_gb:.2f} GB")
    else:
        max_memory_gb = 0
        print(f"\nTraining Time: {training_time:.1f}s")
        
    print("="*50)
    print(f"Process complete. Best model is saved in {output_dir}")
    print("\n")
        

if __name__ == '__main__':
    main()