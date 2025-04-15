# 导入必要的库和模块
import os
import platform
import argparse  # 用于解析命令行参数
import time
import math
import warnings

import pandas as pd
import torch  # PyTorch深度学习框架
import torch.nn.functional as F
import torch.distributed as dist  # 分布式训练支持
from contextlib import nullcontext

from torch import optim, nn  # 优化器和神经网络模块
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行训练
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载器
from transformers import AutoTokenizer, AutoModelForCausalLM  # Hugging Face transformers库
from model.model import MiniMindLM  # 自定义的MiniMind语言模型
from model.LMConfig import LMConfig  # 语言模型配置
from model.dataset import SFTDataset  # SFT数据集类

# 忽略警告信息
warnings.filterwarnings('ignore')


# 日志记录函数，在分布式训练中只在主进程(rank=0)打印日志
def Logger(content):
    if not ddp or dist.get_rank() == 0:  # 如果不是分布式训练或者是主进程
        print(content)


# 学习率调整函数 - 余弦退火学习率调度
# 随着训练进行，学习率会从初始值逐渐降低，帮助模型更好地收敛
def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


# 训练一个完整的epoch
def train_epoch(epoch, wandb):
    # 定义损失函数：交叉熵损失，不进行reduction以便后续使用loss_mask
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    # 遍历数据加载器中的每个批次
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 将数据移动到指定设备(GPU/CPU)
        X = X.to(args.device)  # 输入序列
        Y = Y.to(args.device)  # 目标序列
        loss_mask = loss_mask.to(args.device)  # 损失掩码，用于忽略padding等位置
        # 根据当前步骤计算学习率（余弦退火）
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        # 更新优化器中的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使用自动混合精度上下文（在GPU上可加速训练并减少内存使用）
        with ctx:
            # 前向传播
            res = model(X)
            # 计算损失：预测logits与目标标签之间的交叉熵
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),  # 将logits展平
                Y.view(-1)  # 将目标标签展平
            ).view(Y.size())

            # 应用损失掩码并计算平均损失
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            # 添加辅助损失（如果有的话，例如MoE的负载均衡损失）
            loss += res.aux_loss
            # 如果使用梯度累积，则将损失除以累积步数
            loss = loss / args.accumulation_steps

        # 反向传播：计算梯度（使用梯度缩放器以支持混合精度训练）
        scaler.scale(loss).backward()

        # 梯度累积：每accumulation_steps步才更新一次参数
        if (step + 1) % args.accumulation_steps == 0:
            # 将梯度从FP16反缩放回FP32
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 使用优化器更新模型参数
            scaler.step(optimizer)
            # 更新梯度缩放器
            scaler.update()

            # 清零梯度，准备下一次计算
            optimizer.zero_grad(set_to_none=True)

        # 定期打印训练日志
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            # 如果启用了wandb且是主进程，则记录指标
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 定期保存模型检查点
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()  # 切换到评估模式
            # 根据是否使用MoE（混合专家模型）设置文件名
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/full_sft_{lm_config.dim}{moe_path}.pth'

            # 获取模型状态字典（对于DDP模型需要获取.module属性）
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 保存模型权重
            torch.save(state_dict, ckp)
            model.train()  # 切换回训练模式


# 初始化模型和分词器
def init_model(lm_config):
    # 加载预训练的tokenizer（用于将文本转换为token ID）
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    # 创建MiniMindLM模型实例
    model = MiniMindLM(lm_config)
    # 根据是否使用MoE确定预训练模型路径
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'./out/pretrain_{lm_config.dim}{moe_path}.pth'
    print(f"加载权重{ckp}")
    # 加载预训练模型权重
    state_dict = torch.load(ckp, map_location=args.device)
    # 将权重加载到模型中（strict=False允许部分权重不匹配）
    model.load_state_dict(state_dict, strict=False)
    # 打印模型参数量
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    # 将模型移动到指定设备
    model = model.to(args.device)
    return model, tokenizer


# 初始化分布式训练环境
def init_distributed_mode():
    if not ddp: return  # 如果不是分布式训练则直接返回
    global ddp_local_rank, DEVICE

    # 初始化分布式进程组，使用NCCL后端（适用于GPU训练）
    dist.init_process_group(backend="nccl")
    # 获取当前进程的全局排名
    ddp_rank = int(os.environ["RANK"])
    # 获取当前进程在本机的局部排名
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    # 获取总进程数
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    # 设置当前进程使用的GPU设备
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# 主程序入口
if __name__ == "__main__":
    # 定义命令行参数解析器
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    # 输出目录
    parser.add_argument("--out_dir", type=str, default="out")
    # 训练轮数
    parser.add_argument("--epochs", type=int, default=1)
    # 批次大小
    parser.add_argument("--batch_size", type=int, default=32)
    # 学习率
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    # 训练设备（GPU或CPU）
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    # 数据类型（float16/bfloat16可启用混合精度训练）
    parser.add_argument("--dtype", type=str, default="bfloat16")
    # 是否使用wandb记录训练过程
    parser.add_argument("--use_wandb", action="store_true")
    # wandb项目名称
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")
    # 数据加载器的工作线程数
    parser.add_argument("--num_workers", type=int, default=1)
    # 是否使用分布式训练
    parser.add_argument("--ddp", action="store_true")
    # 梯度累积步数（可用于模拟更大的批次大小）
    parser.add_argument("--accumulation_steps", type=int, default=1)
    # 梯度裁剪阈值
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # 预热迭代次数
    parser.add_argument("--warmup_iters", type=int, default=0)
    # 日志打印间隔
    parser.add_argument("--log_interval", type=int, default=100)
    # 模型保存间隔
    parser.add_argument("--save_interval", type=int, default=100)
    # 本地进程排名（分布式训练用）
    parser.add_argument('--local_rank', type=int, default=-1)
    # 模型隐藏层维度
    parser.add_argument('--dim', default=512, type=int)
    # 模型层数
    parser.add_argument('--n_layers', default=8, type=int)
    # 最大序列长度
    parser.add_argument('--max_seq_len', default=512, type=int)
    # 是否使用混合专家模型(MoE)
    parser.add_argument('--use_moe', default=False, type=bool)
    # 训练数据路径
    parser.add_argument("--data_path", type=str, default="./dataset/sft_mini_512.jsonl")

    # 解析命令行参数
    args = parser.parse_args()

    # 创建语言模型配置
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    # 设置保存目录
    args.save_dir = os.path.join(args.out_dir)
    # 创建必要的目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    # 计算每次迭代处理的token数量
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    # 确定设备类型
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 设置wandb运行名称
    args.wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 创建自动混合精度上下文（在GPU上使用，可加速训练）
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    # 检测是否为分布式训练环境
    ddp = int(os.environ.get("RANK", -1)) != -1  # 是否为分布式训练？
    # 初始化分布式训练相关变量
    ddp_local_rank, DEVICE = 0, "cuda:0"
    # 设置随机种子，确保可复现性
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    # 如果是分布式训练，初始化分布式环境
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        # 为不同进程设置不同的随机种子
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    # 初始化wandb（仅在主进程中）
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config)

    # 创建SFT数据集
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    # 如果是分布式训练，使用DistributedSampler确保数据正确分片
    train_sampler = DistributedSampler(train_ds) if ddp else None
    # 创建数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,  # 批次大小
        pin_memory=True,  # 将数据固定在内存中，加速GPU访问
        drop_last=False,  # 不丢弃最后一个不完整的批次
        shuffle=False,  # 不打乱数据顺序（分布式训练由sampler控制）
        num_workers=args.num_workers,  # 数据加载的工作线程数
        sampler=train_sampler  # 数据采样器
    )

    # 创建梯度缩放器（用于混合精度训练）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    # 创建AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 如果是分布式训练，将模型包装为DistributedDataParallel
    if ddp:
        # 设置不参与同步的参数（通常是位置编码等）
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 计算每个epoch的迭代次数
    iter_per_epoch = len(train_loader)
    # 开始训练循环
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
